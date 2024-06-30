# 导入Flask的Blueprint
from flask import Blueprint, jsonify, request
from flask_cors import CORS

from app.common.response_wrapper import ResponseWrapper

# 创建一个名为post_blueprint的Blueprint实例
post_blueprint = Blueprint('post', __name__)
CORS(post_blueprint)


@post_blueprint.route('/post_example', methods=['POST'])
def post_example():
    data = request.json
    return jsonify(ResponseWrapper.success().data(data).to_dict())
