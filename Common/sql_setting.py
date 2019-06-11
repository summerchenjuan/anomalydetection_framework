#数据库配置


# NAME = 'anomalydecetion'  # 数据库名称
NAME = 'anomalydetection'
USER = 'root'  # 链接数据库的用户名
PASSWORD = '123456'  # 链接数据库的密码
# HOST = '127.0.0.1'  # mysql服务器的域名和ip地址
HOST = '192.168.83.18'
PORT = '3306'  # mysql的一个端口号,默认是3306

DATABASE_URL = 'mysql://{user}:{password}@{host}:{port}/{name}'.format(
    user=USER,
    password=PASSWORD,
    host = HOST,
    port = PORT,
    name = NAME,
)