---
name: script-sandbox-demo
description: "脚本沙箱演示技能：演示参数Schema校验、类型自动转换、命令注入防护、敏感路径拦截、网络禁用"
version: "1.0.0"
author: fast_agents
license: MIT
tags: [demo, sandbox, security]
---

# 脚本沙箱演示技能

本技能演示【C18方案B】之后的脚本沙箱能力，注意与旧写法的两点不同：

## 一、一级元数据已回归极简

frontmatter 只保留 name / description / version / author / license / tags，
**不再写 `scripts:` 段，也不再写 `allow_network` / `allow_write_paths` / `env_whitelist`**：

- **脚本参数 Schema 外置**：放在脚本同目录的 `scripts/<脚本名>.schema.json`，
  和同名 `.py` 放一起，改脚本顺手改 schema；
- **沙箱策略上移到宿主层**：由 `SkillRegistry` / `SkillLoader` / `SkillMiddleware`
  初始化参数（`default_allow_network` / `default_allow_write_paths` / `default_env_whitelist`）
  统一配置，宿主可统一封顶与审计。

## 二、可用脚本

### greet.py
根据姓名和年龄打招呼，并打印沙箱环境信息（验证断网与类型转换）。
参数契约见同目录 `greet.schema.json`。

### export_report.py
把报告导出到技能自身的 `tmp/` 目录（`tmp/` 由宿主自动加入可写白名单）。
参数契约见同目录 `export_report.schema.json`。

## 三、演示的能力

1. 参数 Schema 外置（`scripts/*.schema.json`）并在执行前自动校验
2. int / str / bool / enum 类型自动转换
3. 必填参数、类型、数值范围、枚举值的校验
4. 命令注入字符自动拦截
5. 敏感路径访问拦截
6. 网络访问禁用（宿主层全局默认，脚本内可见 `SKILL_SANDBOX_NETWORK_DISABLED=1`）
