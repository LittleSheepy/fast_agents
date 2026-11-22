---
name: demo-skill
description: C14技能热重载演示技能，修改本文件后热重载会自动检测
version: 1.0.0
tags: [demo, hot-reload]
allowed_tools: ["enable_skill_hot_reload", "reload_skills_now"]
---

# 热重载演示技能 v1.0.0

## 功能说明
这是一个用于演示C14技能热重载功能的示例技能。
修改本SKILL.md文件、scripts/目录下的脚本、references/目录下的参考文档后，
启用热重载的框架会自动检测变更并重新加载技能，无需重启Agent。

## 使用步骤
1. 先调用 enable_skill_hot_reload(interval=2) 启用热重载
2. 修改当前SKILL.md的版本号为v1.0.1，保存文件
3. 等待2秒，框架会自动检测到修改并卸载当前已加载版本
4. 下次调用 load_skill("hot-reload-demo") 时会加载新版本内容

## 可用脚本
- greet.py: 打印欢迎信息
