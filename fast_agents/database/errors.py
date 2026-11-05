"""Database 模块异常体系（基类 + 具体子类，关系型 / 向量型后端共用）。

参考主流框架：
- SQLAlchemy Core 的 ``DBAPIError`` 层级——把驱动异常拆成 ConnectionError /
  OperationalError / IntegrityError / ProgrammingError 等，供上层按「要不要重试 /
  要不要提示用户」分流；
- 同仓库 ``mcp/errors.py``、``tools/tool_exception.py``——「基类 + 具体子类 +
  集中放在模块自己的 errors.py」的组织方式。

设计要点：
- :class:`DatabaseError` **刻意继承 ``ValueError``**：历史上 SQLiteClient 把底层
  ``sqlite3.Error`` 一律包成 ``ValueError`` 抛出，既有调用方（如 memory 层
  ``except (ValueError, TypeError)``）与测试断言（``pytest.raises(ValueError)``）
  都依赖 ``ValueError`` 谱系。沿用该谱系可做到**零破坏**——上层既可用新的具体类型
  精确分流，也可继续用 ``except ValueError`` 兜底。
- 本模块的异常是「把底层驱动异常 + 入参校验错误统一收口」的结果，**不是**把底层库
  原生异常直接抛给上层（那样上层就得 ``import sqlite3``，破坏分层）。
- 只覆盖「数据库自身 / 底层驱动 / 入参」产生的错误；调用方把类型传错（如把 int 当
  表名）仍抛原生 ``TypeError``，不纳入本体系。
"""
from __future__ import annotations


class DatabaseError(ValueError):
    """Database 模块异常基类。

    刻意继承 ``ValueError`` 以兼容既有的 ``except ValueError`` 与
    ``pytest.raises(ValueError)``；新代码建议直接捕获本类或其具体子类。
    """


class DatabaseConnectionError(DatabaseError):
    """连接层：建立 / 关闭数据库连接失败（路径不存在、文件不可写、驱动不可用等）。"""


class DatabaseOperationalError(DatabaseError):
    """执行层：SQL 执行 / 查询失败（语法错误、约束冲突、锁等待超时等）。"""


class DatabaseValidationError(DatabaseError):
    """校验层：入参不合法（列定义非法、必填缺失、危险操作被拒等）。"""
