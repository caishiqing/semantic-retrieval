# semantic-encoder

## 简介

将文本编码成语义向量，相似的文本具有更高的相似度。

模型介绍：[sentence-transformers](https://www.sbert.net/docs/pretrained_models.html)

文本向量使用[vearch](https://github.com/vearch/vearch/)维护。

 

## 配置

在`config.yml`中完成参数配置：

- elasticsearch：ES引擎配置
  - cluster：集群，服务器列表
  - index：索引名称
  - properties：需要创建向量的字段，向量字段的命名规则为在原字段后面添加 "_vector"
  - query_batch_size：批量更新向量时的批处理大小
  - timeout：连接超时时间
  - max_retries：连接重启次数
- vearch：vearch向量搜索引擎配置
  - master：master 服务器地址
  - router：router 服务器地址
  - db_name：库名
- model：语义模型配置
  - name：模型名称
  - use_cuda：是否使用 GPU（暂时没用）
  - batch_size：批处理大小（暂时没用）

 

## 依赖

```shell
pip install -r requirements.txt
```

 

## 使用

先完成相关配置

### 向量批量更新

批量更新 ES 向量索引，如果指定向量字段不存在则自动创建：

```shell
python update.py --cover=False
```

`cover`参数表示当向量已经存在时，是否重新计算并覆盖，默认否。

启动脚本后，程序会自动扫描库表，对向量字段为空的记录计算原始字段（如`product_nm`）的向量表征，并更新对应的向量字段（对应`product_nm_vector`）。

 

### 启动向量服务

启动向量计算 web 服务：

```shell
uvicorn server:app --host 0.0.0.0 --port 27018 --reload
```

根据需求替换端口号。

启动后可查看对应的接口文档：{host}:{port}/docs



## Docker

制作镜像：

```shell
docker build -it semantic-encoder .
```

启动容器：

```shell
docker run -p 27018:27018 -itd semantic-encoder
```

