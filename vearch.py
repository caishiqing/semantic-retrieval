from typing import List, Union
import requests
import logging
import hashlib
import os


class Vearch:
    def __init__(self,
                 master: str = "http://127.0.0.1:8817",
                 router: str = "http://127.0.0.1:9001",
                 db_name: str = "test_db"):

        self.master = master
        self.router = router
        self.db_name = db_name

        self.logger = logging.Logger("vearch", level=logging.INFO)
        formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(formater)
        f_handler = logging.FileHandler("vearch.log")
        f_handler.setFormatter(formater)
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

        res = requests.get(os.path.join(master, "list", "db")).json()
        if res["code"] != 200:
            self.logger.warning(res["msg"])
            return

        if self.db_name not in [db["name"] for db in res["data"]]:
            res = requests.put(
                os.path.join(self.master, "db", "_create"),
                json={"name": self.db_name}).json()
            self._log(res, "Create dataset", self.db_name)

    def delete(self):
        spaces = requests.get(
            url=os.path.join(self.master, "list", "space"),
            params={"db": self.db_name}
        ).json()["data"]
        for space in spaces:
            self.delete_space(space["name"])

        res = requests.delete(os.path.join(self.master, "db", self.db_name)).json()
        self._log(res, "Delete database", self.db_name)
        return res

    def check(self):
        res = requests.get(os.path.join(self.master, "db", self.db_name)).json()
        self._log(res, "Check database", self.db_name)
        return res

    def create_space(self, name: str,
                     index_size: int = 8192,
                     dimensions: int = 768,
                     ncentroids: int = 2048,
                     ):
        param = {
            "name": name,
            "partition_num": 1,
            "replica_num": 1,
            "engine": {
                "name": "gamma",
                "index_size": index_size,
                "id_type": "String",
                "retrieval_type": "IVFPQ",
                "retrieval_param": {
                    "metric_type": "InnerProduct",
                    "ncentroids": ncentroids,
                    "nsubvector": 2
                }
            },
            "properties": {
                "text": {
                    "type": "keyword"
                },
                "embed": {
                    "type": "vector",
                    "dimension": dimensions,
                    "format": "normalization",
                    "store_type": "RocksDB",
                    "store_param": {
                        "cache_size": 512,
                        "compress": False
                    }
                }
            }
        }

        url = os.path.join(self.master, "space", self.db_name, "_create")
        res = requests.put(url, json=param).json()
        return self._log(res, "Create space", name)

    def check_space(self, name: str):
        url = os.path.join(self.master, "space", self.db_name, name)
        res = requests.get(url).json()

        self._log(res, "Check space", name)
        return res

    def delete_space(self, name: str):
        url = os.path.join(self.master, "space", self.db_name, name)
        res = requests.delete(url).json()

        self._log(res, "Delete space", name)
        return res

    def insert_data(self, space_name: str,
                    text: str,
                    embed: List[float],
                    id: Union[str, int] = None):
        if id is None:
            id = hashlib.md5(text.encode()).hexdigest()

        param = {"text": text, "embed": {"feature": embed}}
        url = os.path.join(self.router, self.db_name, space_name, str(id))
        res = requests.post(url, json=param).json()

        if not self._log(res, "Insert data", str(id)):
            return

        return res["_id"]

    def update_data(self, space_name: str,
                    id: Union[str, int],
                    text: str = None,
                    embed: List[float] = None):

        if text is None and embed is None:
            self.logger.warning("Update warning: text and embed can not be both null!")

        param = {"text": text, "embed": embed}
        if text is None:
            param.pop("text")
        if embed is None:
            param.pop("embed")

        url = os.path.join(self.router, self.db_name, space_name, str(id), "_update")
        res = requests.post(url, json=param).json()

        if not self._log(res, "Update data", str(id)):
            return

        return res["_id"]

    def query_by_id(self, space_name: str, id: Union[str, int]):
        url = os.path.join(self.router, self.db_name, space_name, str(id))
        res = requests.get(url).json()
        if not self._log(res, "Query by id", id):
            return

        result = {
            "_id": res["_id"],
            "text": res["_source"]["text"],
            "vector": res["_source"]["embed"]["feature"]
        }
        return result

    def query_by_ids(self, space_name: str, ids: List[Union[str, int]]):
        param = {
            "query": {
                "fields": ["text", "embed"],
                "ids": ids
            },
        }
        url = os.path.join(self.router, self.db_name, space_name, "_query_byids")
        res = requests.post(url, json=param).json()

        if not self._log(res, "Query data by ids", ''):
            return

        if len(ids) == 1:
            res = [res]

        results = []
        for r in res:
            if r["found"]:
                results.append(
                    {
                        "_id": r["_id"],
                        "text": r["_source"]["text"],
                        "vector": r["_source"]["embed"]["feature"]
                    }
                )
            else:
                results.append({})

        return results

    def search(self, space_name: str,
               embed: List[float],
               topk: int,
               nprobe: int = 20,
               return_vector: bool = True):

        param = {
            "query": {
                "sum": [{
                    "field": "embed",
                    "feature": embed,
                    "min_score": 0.0,
                    "max_score": 2.0,
                    "boost": 1.0
                }],
                "filter": []
            },
            "retrieval_param": {
                "nprobe": nprobe
            },
            "fields": ["text", "embed"],
            "is_brute_search": 0,
            "online_log_level": "debug",
            "quick": False,
            "vector_value": False,
            "client_type": "leader",
            "l2_sqrt": False,
            "sort": [{"embed": {"order": "asc"}}],
            "size": topk
        }
        url = os.path.join(self.router, self.db_name, space_name, "_search")
        res = requests.post(url, json=param).json()

        if not self._log(res, "Search", ''):
            return

        results = []
        for hit in res["hits"]["hits"]:
            item = {
                "_id": hit["_id"],
                "text": hit["_source"]["text"],
                "score": (1 - hit["_score"] / 2),
            }
            if return_vector:
                item["vector"] = hit["_source"]["embed"]["feature"]
            results.append(item)

        return results

    def delete_data(self, space_name: str, id: Union[str, int]):
        url = os.path.join(self.router, self.db_name, space_name, str(id))
        res = requests.delete(url)
        return self._log(res, "Delete data", id)

    def _log(self, result: dict, action: str, name: str):
        if not isinstance(result, dict):
            return 1

        if "error" in result:
            self.logger.error("{} {} failed: {}".format(action, name, result["error"]["reason"]))
            return 0

        status_code = result.get("code") or result.get("status") or 200
        if status_code != 200:
            msg = result.get("msg") or "failed"
            self.logger.warning("{} {}: {}".format(action, name, msg))
        else:
            msg = result.get("msg") or "success"
            self.logger.info("{} {}: {}".format(action, name, msg))

        return 1


if __name__ == "__main__":
    vearch = Vearch()
    vearch.check()
    vearch.create_space("aaa", dimensions=4)
    vearch.check_space("aaa")
    vearch.insert_data("aaa", "Hello world!", embed=[0.1, 0.2, 0.3, 0.4], id="123")
    vearch.update_data("aaa", "123", embed=[0.1, 0.2, 0.3, 0.5])
    vearch.insert_data("aaa", "你好", embed=[-0.1, 0.2, -0.3, 0.4], id="456")
    result = vearch.query_by_id("aaa", id="123")
    print("Query result: ", result)
    vearch.query_by_ids("aaa", ids=["123", "456", "678"])
    result = vearch.search("aaa", [0.1, 0.2, 0.3, 0.5], topk=10)
    print("Search result: ", result)
    vearch.delete_space("aaa")
    vearch.delete()
