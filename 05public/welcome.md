---
title: 欢迎来到我的博客
description: 这是第一篇示例文章，演示 frontmatter 写法和 Obsidian 双链。
pubDate: 2026-10-01
tags: [meta, 随笔]
---

## 这篇文章是干嘛的

这是放在 `05public/` 里的第一篇示例，验证整个管线能跑通：
Obsidian 写 Markdown → Astro 读取 → 构建成静态页面。

你可以在 Obsidian 里直接编辑它，也可以直接删掉。

## 支持的写法

### 1. Frontmatter

每篇文章顶部用 YAML 写元信息：

```yaml
---
title: 文章标题
description: 一句话摘要
pubDate: 2026-10-01
tags: [Java, Web3]
draft: false   # true 则不发布
---
```

### 2. Obsidian 双链

写 `[[Transformer QKV]]` 会被转成链接，指向 `/blog/transformer-qkv/`。
也支持别名 `[[目标笔记|显示文字]]`。

**试试下面这条真的双链**（已建好对应的 `Transformer QKV.md`）：

- [[Transformer QKV]] ← 普通双链
- [[Transformer QKV|点我看 QKV 详解]] ← 带别名，显示自定义文字

> 注意：链接的目标 slug 来自文件名。如果你在 Obsidian 里用 `[[某篇笔记]]`，
> 那篇笔记必须也在 `05public/` 里，且文件名 slug 化后能对上。

### 3. 代码块

```python
def hello(name: str) -> str:
    return f"Hello, {name}!"
```

### 4. 引用和列表

- 第一项
- 第二项
  - 嵌套

> 引用块长这样。

---

写完后 `git push` 即可自动部署（部署配置见 README）。
