# 示例配置

本目录只保存可提交的路径写法和配置说明，不保存私人机器的数据位置。

direct-run wrapper 使用以下仓库相对 placeholder：

```text
data/example_input.tif
data/example_capture_PA1.bin
outputs/denoise_example/
```

运行前可在本地编辑 wrapper 顶部配置，或更推荐通过 CLI 传入绝对/相对路径。
不要把 `/mnt/...`、用户 home 目录或真实大数据路径作为仓库默认值提交。
