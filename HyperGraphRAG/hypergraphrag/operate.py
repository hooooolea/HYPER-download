import asyncio
import json
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 50.0
    )
    hyper_relation = now_hyper_relation
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        weight=weight,
        hyper_relation=hyper_relation,
        source_id=entity_source_id,
    )


async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"hyper-relation"':
        return None
    # add this record as edge
    knowledge_fragment = clean_str(record_attributes[1])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        hyper_relation="<hyperedge>"+knowledge_fragment,
        weight=weight,
        source_id=edge_source_id,
    )
    

async def _merge_hyperedges_then_upsert(
    hyperedge_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []

    already_hyperedge = await knowledge_graph_inst.get_node(hyperedge_name)
    if already_hyperedge is not None:
        already_weights.append(already_hyperedge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_hyperedge["source_id"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    node_data = dict(
        role = "hyperedge",
        weight=weight,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        hyperedge_name,
        node_data=node_data,
    )
    node_data["hyperedge_name"] = hyperedge_name
    return node_data


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        role="entity",
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_data = []
    
    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]
        
        already_weights = []
        already_source_ids = []
        
        if await knowledge_graph_inst.has_edge(hyper_relation, entity_name):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_name)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
        
        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(
            set([source_id] + already_source_ids)
        )

        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_name,
            edge_data=dict(
                weight=weight,
                source_id=source_id,
            ),
        )

        edge_data.append(dict(
            src_id=hyper_relation,
            tgt_id=entity_name,
            weight=weight,
        ))

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    concept_vdb: BaseVectorStorage,
    relations_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    Phase 3 重写版：使用 JSON 格式提取 L1 Concept + 关系，
    然后为每个 chunk 创建 L2 KnowledgeFragment 节点，
    最后建立 Concept → Fragment 的 Contains 边。

    去重规则：Concept 只按 exact name 精确合并，不设相似度阈值。
    """
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["extract_concepts_and_relations_examples"]):
        examples = "\n".join(
            PROMPTS["extract_concepts_and_relations_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["extract_concepts_and_relations_examples"])

    entity_extract_prompt = PROMPTS["extract_concepts_and_relations"]
    context_base = dict(
        language=language,
        examples=examples,
    )

    async def _call_extract_prompt(chunk_key: str, content: str) -> dict:
        """调用 extract_concepts_and_relations prompt，返回解析后的 dict"""
        prompt = entity_extract_prompt.format(**context_base, input_text=content)
        history = pack_user_ass_to_openai_messages(prompt, "")

        final_result = await use_llm_func(prompt)
        history += pack_user_ass_to_openai_messages(prompt, final_result)

        for glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(
                PROMPTS.get("entiti_continue_extraction", ""),
                history_messages=history,
            )
            history += pack_user_ass_to_openai_messages(
                PROMPTS.get("entiti_continue_extraction", ""), glean_result
            )
            final_result += glean_result
            if glean_index == entity_extract_max_gleaning - 1:
                break
            if_loop_result = await use_llm_func(
                PROMPTS.get("entiti_if_loop_extraction", ""),
                history_messages=history,
            )
            if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
                break

        try:
            # 去掉 markdown code fence（如有）
            cleaned = final_result.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for chunk {chunk_key}: {e}")
            return {"concepts": [], "relations": []}

    # ============================================================
    # Phase 1：并行抽取所有 chunk 的 Concept + Relation
    # ============================================================
    logger.info("[Entity Extraction] Extracting concepts and relations...")
    chunk_results: list[dict] = []
    for idx, (chunk_key, chunk_dp) in enumerate(ordered_chunks):
        result = await _call_extract_prompt(chunk_key, chunk_dp["content"])
        chunk_results.append({
            "chunk_key": chunk_key,
            "chunk_dp": chunk_dp,
            "concepts": result.get("concepts", []),
            "relations": result.get("relations", []),
        })
        ticks = PROMPTS["process_tickers"][idx % len(PROMPTS["process_tickers"])]
        print(
            f"{ticks} Processed {idx + 1}/{len(ordered_chunks)} chunks\r",
            end="", flush=True,
        )

    # ============================================================
    # Phase 2：全量 Concept exact-name 去重合并
    # ============================================================
    # concept_name → {domain: set, description: list, source_chunks: set}
    concept_map: dict[str, dict] = {}
    for cr in chunk_results:
        for concept in cr["concepts"]:
            name = concept.get("name", "").strip()
            if not name:
                continue
            # exact-name 精确匹配，大小写归一
            name_key = name.upper()
            if name_key not in concept_map:
                concept_map[name_key] = {
                    "name": name,
                    "domain_set": set(concept.get("domain", [])),
                    "description_list": [],
                    "source_chunks": set(),
                }
            concept_map[name_key]["domain_set"].update(concept.get("domain", []))
            if concept.get("description"):
                concept_map[name_key]["description_list"].append(concept["description"])
            concept_map[name_key]["source_chunks"].add(cr["chunk_key"])

    logger.info(f"[Entity Extraction] {len(concept_map)} unique concepts after dedup")

    # ============================================================
    # Phase 3：写入 Concept 节点到 KG（role="concept"）
    # ============================================================
    all_concepts_data = []
    for name_key, info in concept_map.items():
        description = GRAPH_FIELD_SEP.join(
            sorted(set(info["description_list"]))
        ) if info["description_list"] else info["name"]
        # 摘要（如果描述过长）
        description = await _handle_entity_relation_summary(
            info["name"], description, global_config
        )
        source_id = GRAPH_FIELD_SEP.join(info["source_chunks"])
        node_data = dict(
            role="concept",
            name=info["name"],
            domain=list(info["domain_set"]),
            description=description,
            source_id=source_id,
        )
        await knowledge_graph_inst.upsert_node(info["name"], node_data=node_data)
        all_concepts_data.append({
            "entity_name": info["name"],
            "domain": list(info["domain_set"]),
            "description": description,
        })

    # ============================================================
    # Phase 4：为每个 chunk 创建 KnowledgeFragment 节点（role="fragment"）
    # 并建立 Concept → Fragment 的 Contains 边
    # ============================================================
    all_relationships_data = []
    all_fragment_data = []  # fragment 节点数据（用于 fragment_vdb）

    for cr in chunk_results:
        chunk_key = cr["chunk_key"]
        chunk_dp = cr["chunk_dp"]

        # 创建 fragment KF 节点
        fragment_name = f"KF-{chunk_key}"
        fragment_source_id = chunk_key  # 指向自身 chunk
        fragment_node_data = dict(
            role="fragment",
            name=fragment_name,
            content=chunk_dp.get("content", ""),
            # Phase 1 新增字段
            corpus_id=chunk_dp.get("corpus_id", ""),
            source_file=chunk_dp.get("source_file", ""),
            ftype=chunk_dp.get("ftype", ""),
            chapter=chunk_dp.get("chapter", ""),
            domain=chunk_dp.get("domain", []),
            source_id=fragment_source_id,
        )
        await knowledge_graph_inst.upsert_node(fragment_name, node_data=fragment_node_data)
        all_fragment_data.append({
            "fragment_name": fragment_name,
            "content": chunk_dp.get("content", ""),
            "corpus_id": chunk_dp.get("corpus_id", ""),
            "source_file": chunk_dp.get("source_file", ""),
            "ftype": chunk_dp.get("ftype", ""),
            "chapter": chunk_dp.get("chapter", ""),
            "domain": chunk_dp.get("domain", []),
        })

        # 调用 extract_chunk_concept_mapping 建立 chunk→concept 关联
        # 仅当 chunk 有关联 concept 时才建 Contains 边
        concept_names_in_chunk = [c.get("name", "").strip().upper()
                                  for c in cr["concepts"]
                                  if c.get("name", "").strip()]
        for concept_name_upper in concept_names_in_chunk:
            if concept_name_upper not in concept_map:
                continue
            concept_name = concept_map[concept_name_upper]["name"]
            # Contains 边：Concept → Fragment
            edge_key = f"Contains-{concept_name}-{fragment_name}"
            edge_data = dict(
                type="Contains",
                src=concept_name,
                tgt=fragment_name,
                description=f"{concept_name} contains content in chapter {chunk_dp.get('chapter', '')}",
                weight=1.0,
                source_id=chunk_key,
            )
            await knowledge_graph_inst.upsert_edge(
                edge_key,
                concept_name,
                fragment_name,
                edge_data=edge_data,
            )
            all_relationships_data.append({
                "src_id": concept_name,
                "tgt_id": fragment_name,
                "type": "Contains",
                "description": edge_data["description"],
                "keywords": "",
            })

    # ============================================================
    # Phase 5：写入 L1→L1 关系边（Depends / Related / Prerequisite）
    # ============================================================
    for cr in chunk_results:
        chunk_key = cr["chunk_key"]
        for rel in cr["relations"]:
            rel_type = rel.get("type", "")
            if rel_type not in ("Depends", "Related", "Prerequisite"):
                continue  # Contains 已在 Phase 4 处理
            src = rel.get("src", "").strip()
            tgt = rel.get("tgt", "").strip()
            if not src or not tgt:
                continue
            src_upper = src.upper()
            tgt_upper = tgt.upper()
            if src_upper not in concept_map or tgt_upper not in concept_map:
                continue
            src_name = concept_map[src_upper]["name"]
            tgt_name = concept_map[tgt_upper]["name"]
            edge_key = f"{rel_type}-{src_name}-{tgt_name}"
            edge_data = dict(
                type=rel_type,
                src=src_name,
                tgt=tgt_name,
                description=rel.get("description", ""),
                weight=1.0,
                source_id=chunk_key,
            )
            await knowledge_graph_inst.upsert_edge(
                edge_key,
                src_name,
                tgt_name,
                edge_data=edge_data,
            )
            all_relationships_data.append({
                "src_id": src_name,
                "tgt_id": tgt_name,
                "type": rel_type,
                "description": edge_data["description"],
                "keywords": "",
            })

    # ============================================================
    # Phase 6：写入 VDB（concept_vdb + relations_vdb）
    # ============================================================
    if concept_vdb is not None and all_concepts_data:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + " " + " ".join(dp.get("domain", [])) + " " + dp.get("description", ""),
                "entity_name": dp["entity_name"],
            }
            for dp in all_concepts_data
        }
        await concept_vdb.upsert(data_for_vdb)

    if relations_vdb is not None and all_relationships_data:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp.get("type", "") + " " + dp["src_id"] + dp["tgt_id"] + dp.get("description", ""),
            }
            for dp in all_relationships_data
        }
        await relations_vdb.upsert(data_for_vdb)

    if not all_concepts_data and not all_relationships_data:
        logger.warning("Didn't extract any concepts or relations, maybe your LLM is not working")
        return None

    logger.info(
        f"[Entity Extraction] Done: {len(all_concepts_data)} concepts, "
        f"{len(all_relationships_data)} relations, {len(all_fragment_data)} fragments"
    )
    return knowledge_graph_inst


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    concept_vdb: BaseVectorStorage,
    relations_vdb: BaseVectorStorage,
    fragment_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return cached_response
    
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    # Phase 3: 使用新的 JSON prompt 提取查询关键词
    extract_kw_prompt = PROMPTS["extract_concepts_and_relations"]
    kw_context_base = dict(
        language=language,
        examples="",  # 查询时不用示例
    )
    kw_prompt = extract_kw_prompt.format(**kw_context_base, input_text=query)
    final_result = await use_model_func(kw_prompt)

    logger.info("kw_prompt result:")
    print(final_result)

    try:
        cleaned = final_result.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        kw_data = json.loads(cleaned)
        concepts_out = kw_data.get("concepts", [])
        relations_out = kw_data.get("relations", [])
        # ll_keywords: 提取的 concept name 列表（用于 local 模式）
        ll_keywords = [c.get("name", "").strip() for c in concepts_out if c.get("name", "").strip()]
        # hl_keywords: 从关系描述提取（用于 global 模式）
        hl_keywords = [r.get("description", "").strip() for r in relations_out if r.get("description", "").strip()]
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e} {final_result}")
        return PROMPTS["fail_response"]

    # Handle keywords missing
    if not ll_keywords and not hl_keywords:
        logger.warning("low_level_keywords and high_level_keywords are both empty")
        return PROMPTS["fail_response"]
    if not ll_keywords and query_param.mode in ["hybrid", "local"]:
        logger.warning("low_level_keywords is empty")
        return PROMPTS["fail_response"]
    if not hl_keywords and query_param.mode in ["hybrid", "global"]:
        logger.warning("high_level_keywords is empty")
        return PROMPTS["fail_response"]

    # Build context — ll_keywords / hl_keywords 已经是 list，直接传入
    keywords = [ll_keywords, hl_keywords]
    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        concept_vdb,
        relations_vdb,
        fragment_vdb,
        text_chunks_db,
        query_param,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    if query_param.only_need_prompt:
        return sys_prompt
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return response


async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    concept_vdb: BaseVectorStorage,
    relations_vdb: BaseVectorStorage,
    fragment_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):

    ll_kewwords, hl_keywrds = query[0], query[1]
    if query_param.mode in ["local", "hybrid"]:
        if ll_kewwords == "":
            ll_entities_context, ll_relations_context, ll_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn(
                "Low Level context is None. Return empty Low entity/relationship/source"
            )
            query_param.mode = "global"
        else:
            (
                ll_entities_context,
                ll_relations_context,
                ll_text_units_context,
            ) = await _get_node_data(
                ll_kewwords,
                knowledge_graph_inst,
                concept_vdb,
                text_chunks_db,
                query_param,
            )
    if query_param.mode in ["global", "hybrid"]:
        if hl_keywrds == "":
            hl_entities_context, hl_relations_context, hl_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn(
                "High Level context is None. Return empty High entity/relationship/source"
            )
            query_param.mode = "local"
        else:
            (
                hl_entities_context,
                hl_relations_context,
                hl_text_units_context,
            ) = await _get_edge_data(
                hl_keywrds,
                knowledge_graph_inst,
                relations_vdb,
                text_chunks_db,
                query_param,
            )
            if (
                hl_entities_context == ""
                and hl_relations_context == ""
                and hl_text_units_context == ""
            ):
                logger.warn("No high level context found. Switching to local mode.")
                query_param.mode = "local"
    if query_param.mode == "hybrid":
        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    elif query_param.mode == "local":
        entities_context, relations_context, text_units_context = (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        )
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


def _node_has_matching_domain(
    knowledge_graph_inst: BaseGraphStorage,
    node_name: str,
    query_domains: list[str],
) -> bool:
    """
    检查 node 是否属于 query_domains 中的任意一个 domain。
    匹配规则：node.domain ∩ query.domains != ∅
    若 node 无 domain 字段，默认通过（不过滤）。
    注意：此函数为 async caller 准备，实际调用时需 await。
    """
    # 同步接口：直接访问内部 _data 缓存
    node_data = getattr(knowledge_graph_inst, "_data", {}).get(node_name)
    if node_data is None:
        return True  # 查不到的节点默认通过
    node_domains = node_data.get("domain", [])
    if not node_domains:
        return True  # 无 domain 字段的节点默认通过
    return bool(set(node_domains) & set(query_domains))


async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    concept_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # get similar entities (concept_vdb), with optional domain filtering
    results = await concept_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", "", ""

    # Phase 1 新增：domain 过滤
    if query_param.domains:
        results = [
            r for r in results
            if r.get("entity_name") and
               _node_has_matching_domain(
                   knowledge_graph_inst, r["entity_name"], query_param.domains
               )
        ]
    if not results:
        return "", "", ""
    # get entity information
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    # get entity degree
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # get relate edges
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )

    # build prompt
    entites_section_list = [["id", "entity", "type", "description"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "hyperedge", "related_entities"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["description"],
                e["related_nodes"]
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(e)
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, "description": k[1], **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    all_related_nodes = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["src_tgt"][1]) for edge in all_edges_data]
    )
    all_nodes = []
    for this_nodes in all_related_nodes:
        all_nodes.append("|".join([n[1] for n in this_nodes]))
    all_edges_data = [
        {**e, "related_nodes": n}
        for e, n in zip(all_edges_data, all_nodes)
    ]
    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relations_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await relations_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["hyperedge_name"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    # edge_degree = await asyncio.gather(
    #     *[knowledge_graph_inst.node_degree(r["hyperedge_name"]) for r in results]
    # )
    edge_datas = [
        {"hyperedge": k["hyperedge_name"], "rank": k["distance"], **v}
        for k, v in zip(results, edge_datas)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["hyperedge"],
        max_token_size=query_param.max_token_for_global_context,
    )
    all_related_nodes = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["hyperedge"]) for edge in edge_datas]
    )
    all_nodes = []
    for this_nodes in all_related_nodes:
        all_nodes.append("|".join([n[1] for n in this_nodes]))
    edge_datas = [
        {**e, "related_nodes": n}
        for e, n in zip(edge_datas, all_nodes)
    ]

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )

    relations_section_list = [
        ["id", "hyperedge", "related_entities"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["hyperedge"],
                e['related_nodes']
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN")
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["hyperedge"]) for edge in edge_datas]
    )
    
    entity_names = []
    seen = set()

    for node_data in node_datas:
        for e in node_data:
            if e[1] not in seen:
                entity_names.append(e[1])
                seen.add(e[1])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                # Only store valid data
                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources


# ============================================================
# Phase 3 Step 3.3：chunking_by_chapter
# 章节优先分块策略：在 token_size 分块之前，先按章节切分
# 每个章节块内部再按 max_token_size 切分
# ============================================================

def chunking_by_chapter(
    content: str,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
    # Phase 1 新增字段（可选传入，否则从 content 推断）
    corpus_id: str = "",
    source_file: str = "",
    ftype: str = "",
    domain: list[str] | None = None,
    chapter_hint: str = "",
) -> list[TextChunkSchema]:
    """
    章节优先分块：
    1. 若传入 chapter_hint（目录结构 JSON），直接按章节切分
    2. 否则，用 token_size 分块，回退到原有行为
    3. 最终每块附加 chapter / corpus_id / source_file / ftype / domain 字段
    """
    from .utils import (
        encode_string_by_tiktoken,
        decode_tokens_by_tiktoken,
    )

    # ============================================================
    # 分支 A：chapter_hint 提供明确的章节结构
    # chapter_hint 格式：JSON 字符串 [{"chapter": "1", "content": "..."},
    #                                 {"chapter": "1.1", "content": "..."}, ...]
    # ============================================================
    if chapter_hint:
        try:
            chapters_data = json.loads(chapter_hint)
        except json.JSONDecodeError:
            chapters_data = None

        if chapters_data:
            results: list[TextChunkSchema] = []
            global_chunk_index = 0
            for chapter_item in chapters_data:
                chapter_str = chapter_item.get("chapter", "")
                chapter_content = chapter_item.get("content", "")
                if not chapter_content.strip():
                    continue

                # 章节内部按 token_size 再切分
                sub_chunks = _sub_chunking_by_token(
                    chapter_content,
                    overlap_token_size=overlap_token_size,
                    max_token_size=max_token_size,
                    tiktoken_model=tiktoken_model,
                )
                for sub_idx, sub_content in enumerate(sub_chunks):
                    results.append({
                        "tokens": _count_tokens(sub_content, tiktoken_model),
                        "content": sub_content.strip(),
                        "full_doc_id": corpus_id or source_file,
                        "chunk_order_index": global_chunk_index,
                        # Phase 1 新增字段
                        "corpus_id": corpus_id,
                        "source_file": source_file,
                        "ftype": ftype,
                        "chapter": chapter_str,
                        "domain": domain or [],
                    })
                    global_chunk_index += 1
            if results:
                return results

    # ============================================================
    # 分支 B：无 chapter_hint，回退到原有 token_size 分块
    # 同时从 content 中尝试推断 chapter（LLM 推断，保守回退）
    # ============================================================
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results: list[TextChunkSchema] = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start: start + max_token_size], model_name=tiktoken_model
        ).strip()
        if not chunk_content:
            continue
        # 推断当前 chunk 所属的 chapter（从内容特征判断，如 "第X章"）
        inferred_chapter = _infer_chapter_from_content(chunk_content)
        results.append({
            "tokens": min(max_token_size, len(tokens) - start),
            "content": chunk_content,
            "full_doc_id": corpus_id or source_file,
            "chunk_order_index": index,
            # Phase 1 新增字段
            "corpus_id": corpus_id,
            "source_file": source_file,
            "ftype": ftype,
            "chapter": inferred_chapter,
            "domain": domain or [],
        })
    return results


def _sub_chunking_by_token(
    content: str,
    overlap_token_size: int,
    max_token_size: int,
    tiktoken_model: str,
) -> list[str]:
    """章节内容内部按 token_size 再切分"""
    from .utils import (
        encode_string_by_tiktoken,
        decode_tokens_by_tiktoken,
    )
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for start in range(0, len(tokens), max_token_size - overlap_token_size):
        sub = decode_tokens_by_tiktoken(
            tokens[start: start + max_token_size], model_name=tiktoken_model
        ).strip()
        if sub:
            results.append(sub)
    return results


def _count_tokens(content: str, tiktoken_model: str) -> int:
    """计算 content 的 token 数"""
    from .utils import encode_string_by_tiktoken
    return len(encode_string_by_tiktoken(content, model_name=tiktoken_model))


def _infer_chapter_from_content(content: str) -> str:
    """
    保守推断 content 所属的 chapter。
    匹配模式：第X章、第X.1节、第X.Y.Z小节 等。
    若无法推断，返回空字符串。
    """
    import re
    # 优先匹配最具体的路径（如 1.2.3）
    patterns = [
        r"第\s*([0-9]+(?:\.[0-9]+)*)\s*[章节]?",   # "第1.2章" / "第1.2.3节"
        r"^\s*([0-9]+\.[0-9]+(?:\.[0-9]+)*)\s",   # 行首 "1.2.3 "
        r"^第\s*([0-9]+)\s*章",                     # "第1章"
    ]
    for p in patterns:
        m = re.search(p, content, re.MULTILINE)
        if m:
            return m.group(1).strip()
    return ""