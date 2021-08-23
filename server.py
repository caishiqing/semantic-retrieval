from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from typing import Union, List, Dict
from fastapi import FastAPI
from vearch import Vearch
import numpy as np
import yaml


config = yaml.load(open("config.yml"), Loader=yaml.SafeLoader)
model_config, vearch_config = config["model"], config["vearch"]
model = SentenceTransformer(model_config["name"])
vearch = Vearch(**vearch_config)

description = """
SemanticEncoder API helps you encode and search text，for example QA, document search etc.

## Home

For more information please visit [gitlab](https://gitlab.cttq.com/ai/semantic-encoder)

## Use

You well be able to:

- Encode text to semantic vector
- Compute similarity between two texts
- Maintain spaces for different scenes
- Insert, update and delete text data
- Semantic search for txt
"""

app = FastAPI(
    title="SemanticEncoder",
    description=description,
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.get("/encode")
async def encode(text: str) -> dict:
    """将文本编码成向量: {host}:{port}/encode

    请求实例：{"text": "多样屋功夫茶具组合"}

    响应实例：{"vector": [-0.12, 0.34, ..., -0.42]}

    Args:
        text (str): 待编码的文本.

    Returns:
        dict: {"vector": 编码向量}
    """
    vector = model.encode(text, normalize_embeddings=True).tolist()
    return {"vector": vector}


@app.get("/similarity")
async def similarity(text1: str, text2: str) -> dict:
    """计算文本相似度（余弦相似度），不区分顺序: {host}:{port}/similarity

    请求实例：{"text1": "报销单怎么开", "text2": "在哪儿开报销单"}

    响应实例：{"score":0.9187}

    Args:
        text1 (str): 第一个文本;
        text2 (str): 第二个文本.

    Returns:
        dict: {"score": 相似度}
    """
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    score = pytorch_cos_sim(embedding1, embedding2).tolist()[0][0]
    return {"score": score}


@app.put("/create_space/{name}")
async def create_space(name: str) -> str:
    """创建表空间: {host}:{port}/create_space/{name}

    Args:
        name (str): 表空间名称.

    Returns:
        str: 执行结果消息
    """
    res = vearch.create_space(
        name, dimensions=model.get_sentence_embedding_dimension()
    )
    if res:
        return "success"
    return "failed"


@app.get("/check_space/{name}")
async def check_space(name: str) -> dict:
    """查看表空间: {host}:{port}/check_space/{name}

    Args:
        name (str): 表空间名称.

    Returns:
        dict: 表空间信息
    """
    return vearch.check_space(name)


@app.delete("/delete_space/{name}")
async def delete_space(name: str) -> dict:
    """删除表空间: {host}:{port}/delete_space/{name}

    Args:
        name (str): 表空间名称.

    Returns:
        dict: 结果信息
    """
    return vearch.delete_space(name)


@app.post("/insert_data/{space_name}")
async def insert_data(space_name: str, text: str, id: str = None) -> str:
    """插入一条数据: {host}:{port}/insert_data/{space_name}

    请求实例: {"text": "报销单怎么开"}

    响应实例: -5899907996241850837

    Args:
        space_name (str): 表空间名称;
        text (str): 插入的文本;
        id (str, optional): 数据ID（如果为空，则系统生成一个ID）.

    Returns:
        str: 数据ID 或者 null
    """
    vector = model.encode(text, normalize_embeddings=True).tolist()
    res = vearch.insert_data(space_name, text, vector, id)
    return res


@app.post("/update_data/{space_name}")
async def update_data(space_name: str, text: str, id: str) -> str:
    """更新一条数据: {host}:{port}/update_data/{space_name}

    请求实例: {"text": "报销单怎么开"， "id": "-5899907996241850837"}

    响应实例: -5899907996241850837

    Args:
        space_name (str): 表空间名称;
        text (str): 插入的文本;
        id (str): 数据ID（如果为空，则系统生成一个ID）.

    Returns:
        str: 数据ID 或者 null
    """
    vector = model.encode(text, normalize_embeddings=True).tolist()
    res = vearch.update_data(space_name, text=text, embed=vector, id=id)
    return res


@app.get("/query_by_id/{space_name}")
async def query_by_id(space_name: str, id: str) -> dict:
    """根据ID查询: {host}:{port}/query_by_id/{space_name}

    请求实例：{"id": "-5899907996241850837"}

    响应实例: {"_id": "-5899907996241850837", "text": "报销单怎么开", "vector": [0.1, -0.2, ...]}

    Args:
        space_name (str): 表空间名称;
        id (str): 数据ID.

    Returns:
        dict: 数据信息
    """
    return vearch.query_by_id(space_name, id)


@app.get("/search/{space_name}")
async def search(space_name: str,
                 query: str,
                 topk: int = 1,
                 return_vector: bool = False) -> list:
    """语义检索: {host}:{port}/search/{space_name}

    请求实例：{"query": "怎么开具报销单？", "topk": 3}

    响应实例：[{"_id": "-5899907996241850837", "text": "报销单怎么开", "score": 0.9217}, ...]


    Args:
        space_name (str): 表空间名称;
        query (str): 检索文本;
        topk (int, optional): 返回结果数量;
        return_vector (bool, optional): 结果中是否包含向量.

    Returns:
        list: 返回结果
    """
    vector = model.encode(query, normalize_embeddings=True).tolist()
    return vearch.search(space_name, vector, topk, return_vector=return_vector)


@app.delete("/delete_data/{space_name}")
async def delete_data(space_name: str, id: str) -> str:
    """删除数据: {host}:{port}/delete_data/{space_name}

    Args:
        space_name (str): 表空间名称;
        id (str): 数据ID.

    Returns:
        str: 执行结果消息
    """
    res = vearch.delete_data(space_name, id)
    if res:
        return "success"
    return "failed"
