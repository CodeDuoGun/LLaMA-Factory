from app.rag.vdb.es.es_processor import es_handler
from app.model.embedding.tool import get_doubao_embedding


if __name__=="__main__":
    import argparse
    # es_handler = ElasticsearchHandler()
    parser = argparse.ArgumentParser(description="Elasticsearch操作")
    parser.add_argument("--index_name", type=str, help="索引名称")
    parser.add_argument("--operator", type=str, default="update", help="进行的数据库操作")
    # parser.add_argument("--alpha", type=bool, default=True, help="是否为本地使用")

    args = parser.parse_args()
    if args.operator == "update":
        if args.index_name == "qa":
            q1 = "广州中医门诊能用医保么？"
            a1 = """您好！广州四惠中医门诊部是医保定点单位，您在该门诊就诊的费用是可以医保报销的。"""
            qa_data = {q1: a1}
            for envir in ("alpha", "prod"):
                index_name = f"{envir}_{args.index_name}"
                # es_handler.delete_qa(qa_data, index_name)
                es_handler.update_qa(qa_data, index_name)

        elif args.index_name == "doctor":
            # 更新doctor
            doctor_data = {
                1256:
                    {
                        "所在区域": "北京",
                        "序号": "",
                        "姓名":"王学川",
                        "简介":"广州医科大学附属中医医院，脾胃科副主任医师，硕士研究生导师。师从国家级重点专科脾胃病科、重点学科脾胃消化学科学术带头人许鑫梅教授 及广东省名中医邝卫红教授。",
                        "ID":"1256",
                        "擅长":"萎缩性胃炎、慢性胃炎、肠上皮化生、胃糜烂、胃溃疡、急慢性肠炎、肠息肉、胃息肉等各类胃肠道疾病。",
                        "出诊地点":"北京四惠医疗互联网医院",
                        "执业医院":"北京四惠医疗互联网医院，广州医科大学附属中医医院",
                        "特殊称号/职称":"副主任医师"
                    },
            }
            for envir in ("alpha", "prod"):
                index_name = f"{envir}_{args.index_name}"
                es_handler.update_doctor(doctor_data, index_name)
        elif "disease" in args.index_name:
            index = f"alpha_{args.index_name}"
            data = [{"专家姓名": "马东来", "疾病种类": "皮肤"}]

            es_handler.update_disease(index, data)

    elif args.operator == "delete":
        """
        
        """
        if args.index_name == "qa":
            qs = [
            ]
            for q in qs:
                a1 = """ """
                qa_data = {q.strip(): a1}
                for envir in ("alpha", "prod"):
                    index_name = f"{envir}_{args.index_name}"
                    es_handler.delete_qa(qa_data, index_name)

        elif args.index_name == "doctor":
            doctor_ids = ["1256"]
            for envir in ("alpha", "prod"):
                index_name = f"{envir}_{args.index_name}"
                es_handler.delete_doctor(doctor_ids, index_name)

        elif "disease" in args.index_name:
            index = f"alpha_{args.index_name}"
            data = [{"专家姓名": "马东来", "疾病种类": "皮肤"}]

            es_handler.delete_disease(index, data)

    elif args.operator == "search":
        if args.index_name == "doctor":
            # res = es_handler.semantic_search("alpha_doctor", "北京治疗美人鱼综合症的医生", 5)
            # res = es_handler.search_hybrid("alpha_doctor", "", "脱发", k=5)
            index_name =f"alpha_{args.index_name}"
            res = es_handler.search_by_keyword("alpha_doctor", "耳鸣 耳痛", size=5, doctor_location="杭州", doctor_name="")
            # async_res = asyncio.run(es_handler.search_qa_by_answer_async("alpha_qa", "问诊单在哪里填写", top_k=5, embed_model_type="doubao"))
            # async_tmp = [{"question": hit["_source"]["question"], "score": hit['_score']} for hit in async_res]
            # print(async_tmp)
            # res = es_handler.search_by_keyword("alpha_doctor", "", size=5, doctor_location="", doctor_name="王桂绵")
            # tmp = [{"name": hit["_source"]["姓名"], "score": hit['_score'], "区域": hit["_source"]["所在区域"], "hospital":  hit["_source"]["出诊地点"],"good": hit["_source"]["擅长"]} for hit in res]
            # print(tmp)
        elif "disease" in args.index_name:
            index_name = f"alpha_{args.index_name}"
            # res = es_handler.search_by_keyword(index_name, "焦虑症", size=5, doctor_location="", doctor_name="")
            res = es_handler.search_by_vector(f"alpha_primary_disease", "皮肤", top_k=5, doctor_location="")
            tmp = [{"name": hit["_source"]["doctors"], "good": hit["_source"]["secondary_disease"],  "score": hit['_score']} for hit in res] if index_name == "alpha_secondary_disease" else [{"name": hit["_source"]["doctors"], "good": hit["_source"]["primary_disease"], "score": hit['_score']} for hit in res]
            print(tmp)
            
        """多字段关键词匹配"""
