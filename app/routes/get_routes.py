# 导入Flask的Blueprint
from flask import Blueprint, jsonify, request
from flask_cors import CORS

from app.common.response_wrapper import ResponseWrapper
from lstm.lstm_model import predict
from tools.从数据库获取数据 import get_data
from datetime import datetime, timedelta

# 创建一个名为get_blueprint的Blueprint实例
get_blueprint = Blueprint('get', __name__)
CORS(get_blueprint)


@get_blueprint.route('/predictive-statistics', methods=['GET'])
def predictive_statistics():
    statistics = get_data()
    # 就不在后端重新写一个接口了，在这里筛选出今天之前的100天的数据
    today = datetime.today().date()
    hundred_days_ago = today - timedelta(days=100)
    filtered_statistics = [s for s in statistics if hundred_days_ago <= datetime.strptime(s.date, "%Y-%m-%d").date() < today]
    # 按照 date 字段排序
    filtered_statistics.sort(key=lambda x: x.date)
    res = predict(filtered_statistics, model_root_dir="lstm/weight/20240630201053-数据全部持久化", )
    return jsonify(ResponseWrapper.success().set_data(res).to_dict())
