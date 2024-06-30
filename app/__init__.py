# 导入Flask库
from flask import Flask


# 创建Flask应用函数
def create_app():
    app = Flask(__name__)

    # 导入路由模块
    from .routes.get_routes import get_blueprint
    from .routes.post_routes import post_blueprint

    # 注册蓝图
    app.register_blueprint(get_blueprint)
    app.register_blueprint(post_blueprint)

    return app
