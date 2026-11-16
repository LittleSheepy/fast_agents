"""Database 模块异常体系（基类 + 具体子类，关系型 / 向量型后端共用）。

参考主流框架：
- SQLAlchemy Core 的 ``DBAPIError`` 层级——把驱动异常拆成 ConnectionError /
  OperationalError / IntegrityError / ProgrammingError 等，供上层按「要不要重试 /
  要不要提示用户」分流；
- 同仓库 ``mcp/errors.py``、``tools/tool_exception.py``——「基类 + 具体子类 +
  集中放在模块自己的 errors.py」的组织方式。

设计要点：
- :class:`DatabaseError` **继承 ``Exception``**（而非 ``ValueError``）：数据库错误涵盖
  连接失败、锁等待超时、SQL 语法/约束冲突、驱动缺失等，语义上并非「值不合法」，只有
  校验类错误才贴近 ``ValueError``。独立谱系对齐 SQLAlchemy（``SQLAlchemyError(Exception)``）
  与 Django（``DatabaseError(Exception)``）的做法，可避免与无关的 ``except ValueError``
  （如 ``int()`` 转换、``datetime.fromisoformat`` 解析）相互误捕。上层统一用
  ``except DatabaseError`` 或其具体子类分流；``ValueError`` 兜底不再适用。
- 本模块的异常是「把底层驱动异常 + 入参校验错误统一收口」的结果，**不是**把底层库
  原生异常直接抛给上层（那样上层就得 ``import sqlite3``，破坏分层）。
- 只覆盖「数据库自身 / 底层驱动 / 入参」产生的错误；调用方把类型传错（如把 int 当
  表名）仍抛原生 ``TypeError``，不纳入本体系。
"""
from __future__ import annotations


class DatabaseError(Exception):
    """Database 模块异常基类。

    继承 ``Exception``（不挂 ``ValueError``）：数据库错误多为连接 / 执行 / 驱动类问题，
    语义上不是「值不合法」，独立谱系可避免与无关的 ``except ValueError`` 相互误捕。
    上层统一捕获本类或其具体子类来分流。
    """


class DatabaseConnectionError(DatabaseError):
    """连接层：建立 / 关闭数据库连接失败（路径不存在、文件不可写、驱动不可用等）。"""


class DatabaseOperationalError(DatabaseError):
    """执行层：SQL 执行 / 查询失败（语法错误、约束冲突、锁等待超时等）。"""


class DatabaseValidationError(DatabaseError):
    """校验层：入参不合法（列定义非法、必填缺失、危险操作被拒等）。"""


class DatabaseIdentifierError(DatabaseValidationError):
    """校验层：标识符（表名/列名/集合名等）不合法。"""


class DatabaseIntegrityError(DatabaseOperationalError):
    """执行层：约束冲突（主键重复、唯一键冲突、集合已存在等）。"""


class DatabaseQueryError(DatabaseOperationalError):
    """执行层：查询/写入失败（集合不存在、字段不存在等）。"""


class DatabaseDriverMissingError(DatabaseConnectionError):
    """连接层：可选驱动未安装（如pymysql/chromadb未装）。"""
