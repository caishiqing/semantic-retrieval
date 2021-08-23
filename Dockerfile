# 使用pytorch基础镜像
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

COPY ./models/ ./models/
COPY ./* ./

# 镜像操作命令
RUN pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 暴露端口
EXPOSE 27018

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "27018"]