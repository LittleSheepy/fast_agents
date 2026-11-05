"""database 模块 Chroma 向量后端示例。

演示本地持久化模式下的：
- 幂等建集合（get_or_create_collection）
- add：按 id upsert（覆盖写入）
- get：按 id 精确取
- search：向量近邻检索（带 metadata 前置过滤）
- count：全量/按条件计数
- delete：按 id / 按条件删除（空条件被拦截）
- drop_collection：清理集合

运行方式：
    pip install fast-agents[chroma]   # 或 pip install "chromadb>=0.4,<1.0"
    python examples/database/chroma_usage.py

跑完后 ./chroma_demo_data/ 目录会留下持久化数据，删掉即清。
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile

from fast_agents.database import ChromaClient, VectorFilter, VectorFilterOp


def cosine_like_embedding(words_base, text_words, dim=8):
    """一个极简「词袋平均」伪向量，用于演示；真实场景请换成 embedding 模型。"""
    vec = [0.0] * dim
    for i, w in enumerate(words_base):
        if w in text_words:
            vec[i % dim] += 1.0
    # 归一化（cosine 距离对归一化后的向量更直观）
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def main() -> None:
    demo_dir = os.path.join(tempfile.gettempdir(), "fast_agents_chroma_demo")
    shutil.rmtree(demo_dir, ignore_errors=True)

    # 1. 建立本地持久化客户端（首次会自动建目录）
    db = ChromaClient(path=demo_dir)
    print(f"[1] ping: {db.ping()}  数据目录: {demo_dir}")

    # 2. 幂等建集合：集合名 3-63 字符、字母开头、字母/数字/_/-
    db.get_or_create_collection("demo_mem", dimension=8, metric="cosine")
    db.get_or_create_collection("demo_mem", dimension=8, metric="cosine")  # 重复调用幂等
    print(f"[2] collection_exists('demo_mem') = {db.collection_exists('demo_mem')}")
    print(f"    collection_exists('not_exist') = {db.collection_exists('not_exist')}")

    # 3. add：按 id upsert（已存在则覆盖，不存在则新增）
    #    embeddings / documents / metadatas 都可省略，但至少要给一个；列表长度与 ids 一致
    vocab = ["猫", "小鱼干", "睡觉", "颜色", "蓝", "狗", "骨头", "跑"]
    docs = [
        ("m1", "猫喜欢吃小鱼干",       {"kind": "fact",       "topic": "pet"}),
        ("m2", "猫爱睡觉",             {"kind": "fact",       "topic": "pet"}),
        ("m3", "我喜欢蓝颜色",         {"kind": "preference", "topic": "color"}),
        ("m4", "狗爱啃骨头",           {"kind": "fact",       "topic": "pet"}),
        ("m5", "狗跑得很快",           {"kind": "fact",       "topic": "pet"}),
    ]
    ids = [d[0] for d in docs]
    documents = [d[1] for d in docs]
    metadatas = [d[2] for d in docs]
    embeddings = [cosine_like_embedding(vocab, d[1]) for d in docs]
    db.add("demo_mem", ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"[3] add {len(ids)} 条，count = {db.count('demo_mem')}")

    # 4. add 幂等：覆盖 m2 的内容和 metadata，其它条保持不变
    db.add("demo_mem",
           ids=["m2"],
           documents=["猫爱打盹"],
           metadatas=[{"kind": "fact", "topic": "pet", "updated": True}],
           embeddings=[cosine_like_embedding(vocab, "猫爱打盹")])
    m2 = db.get("demo_mem", ids=["m2"])[0]
    print(f"[4] upsert 覆盖后 m2 = {m2['document']!r}  metadata={m2['metadata']}")

    # 5. get：按 id 精确取；include_embeddings=True 才会带向量（默认不带，省内存）
    m3 = db.get("demo_mem", ids=["m3"])[0]
    m3_with_vec = db.get("demo_mem", ids=["m3"], include_embeddings=True)[0]
    print(f"[5] get m3: document={m3['document']!r}  has_embedding={'embedding' in m3} / {'embedding' in m3_with_vec}")

    # 6. search：按单个向量做近邻检索；where 做 metadata 前置过滤
    q_about_cat = cosine_like_embedding(vocab, "猫 小鱼干")
    hits = db.search("demo_mem", q_about_cat, n_results=3)
    print(f"[6] 搜索「猫 小鱼干」 Top3:")
    for h in hits:
        print(f"    - {h['id']:>3}  distance={h['distance']:.4f}  {h['document']}")

    # 6.1 where 过滤：只在 kind=preference 里搜，结果只有 m3
    hits_pref = db.search("demo_mem", q_about_cat, n_results=5,
                          where=VectorFilter(field="kind", op=VectorFilterOp.EQ, value="preference"))
    print(f"[6.1] 带 where(kind=preference) 结果: {[h['id'] for h in hits_pref]}")

    # 6.2 AND 组合过滤：topic=pet 且 kind=fact 且非 preference
    hits_pet = db.search("demo_mem", q_about_cat, n_results=5,
                         where=VectorFilter(op=VectorFilterOp.AND, filters=[
                             VectorFilter(field="topic", op=VectorFilterOp.EQ, value="pet"),
                             VectorFilter(field="kind",  op=VectorFilterOp.EQ, value="fact"),
                         ]))
    print(f"[6.2] AND(topic=pet, kind=fact) 结果: {[h['id'] for h in hits_pet]}")

    # 7. count 带 where：统计 pet 主题的条目数
    pet_count = db.count("demo_mem", where=VectorFilter(field="topic", op=VectorFilterOp.EQ, value="pet"))
    print(f"[7] count(topic=pet) = {pet_count}")

    # 8. delete：空条件被安全拦截（防止误删全量）
    try:
        db.delete("demo_mem")
    except ValueError as e:
        print(f"[8] 空条件 delete 被拦截: {e}")
    # 按 metadata 条件删除：删 preference 类
    removed = db.delete("demo_mem", where=VectorFilter(field="kind", op=VectorFilterOp.EQ, value="preference"))
    print(f"[8] delete where(kind=preference) 影响 {removed} 条，剩余 {db.count('demo_mem')} 条")
    # 按 id 删除
    db.delete("demo_mem", ids=["m5"])
    print(f"[8] delete ids=[m5]，剩余 {db.count('demo_mem')} 条")

    # 9. drop_collection：幂等
    db.drop_collection("demo_mem")
    db.drop_collection("demo_mem")  # 重复删除静默
    print(f"[9] drop 后 collection_exists = {db.collection_exists('demo_mem')}")

    db.close()
    print(f"\n演示完成。数据目录 {demo_dir} 保留，可自行删除。")


if __name__ == "__main__":
    main()
