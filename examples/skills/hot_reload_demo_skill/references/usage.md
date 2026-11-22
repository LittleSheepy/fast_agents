# 热重载使用参考

## 启用热重载
```python
# 在代码中启用
middleware.enable_hot_reload(interval=2.0)
```
或者LLM通过工具调用启用：
`enable_skill_hot_reload(interval=2)`

## 关闭热重载
生产环境部署时建议关闭：`disable_skill_hot_reload()`

## 手动触发重载
不等轮询间隔，立即检查变更：`reload_skills_now()`

## 检测的变更类型
- 新增技能目录 → 自动加入索引
- 修改SKILL.md/scripts/references文件 → 自动重载元数据，已加载则自动卸载
- 删除技能目录 → 自动清理索引和已加载状态
