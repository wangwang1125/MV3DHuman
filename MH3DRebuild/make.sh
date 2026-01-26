#!/bin/bash
#---------------------------------- 需要配置 -----------------------------------#
CONDA_ENV_NAME="python310"
MAIN_PATH="script/main.py"
#------------------------------ END of 需要配置 --------------------------------#

CURRENT_DIR=$(cd `dirname $0`; pwd)
cd $CURRENT_DIR
CONFDATA_DIR=$CURRENT_DIR/confdata
SCRIPT_DIR=$CURRENT_DIR/script
BUILD_PATH=$CURRENT_DIR/build
INSTALL_PATH=$CURRENT_DIR/install
BIN_PATH=$INSTALL_PATH/bin
TEST_OUTPUT=$BUILD_PATH/test_output
PACK_DIR=$INSTALL_PATH/pack

CONDA_PATH=$(conda info -e | grep $CONDA_ENV_NAME | awk '{print $NF}')

cmd=$1
if [ "$cmd" = "clear" ];then
    echo "Start clear"
    rm -rfv "$BUILD_PATH"
    rm -rfv "$INSTALL_PATH"

elif [ "$cmd" = "build" ];then
    echo "Start building"
    echo "1. 创建编译目录"
    mkdir -p "$BUILD_PATH"
    mkdir -p "$INSTALL_PATH"
    mkdir -p "$BIN_PATH"

    echo "2. 使用 stickytape 将 python 脚本合并为一个"
    pip install stickytape
    stickytape "$MAIN_PATH" > "$BUILD_PATH/exec.py"

    echo "3. 使用 py_compile 将脚本转化为 pyc 二进制格式 $CONDA_PATH"
    PYTHON_BIN="${CONDA_PATH}/bin/python"
    if [ ! -x "$PYTHON_BIN" ]; then
        echo -e "\e[31m未找到 conda 环境 python: $PYTHON_BIN\e[0m"
        exit 1
    fi
    "$PYTHON_BIN" -m py_compile "$BUILD_PATH/exec.py"
    exec_py=$(ls "$BUILD_PATH/__pycache__"/exec.*.pyc 2>/dev/null | head -n 1)
    if [ -z "$exec_py" ] || [ ! -f "$exec_py" ]; then
        echo -e "\e[31mpy_compile 未生成 pyc，请检查 exec.py 语法\e[0m"
        exit 1
    fi
    mv "$exec_py" "$BIN_PATH/exec"

elif [ "$cmd" = "test" ];then
    echo "Start testing"
    if [ ! -f "$BIN_PATH/exec" ];then
        echo -e "\e[31m二进制程序 \"$BIN_PATH/exec\"  不存在，请运行 \"$0 build\" 构建！ \e[0m"
        exit 1
    fi
    PYTHON_BIN="${CONDA_PATH}/bin/python"
    echo "版本测试"
    time "$PYTHON_BIN" "$BIN_PATH/exec" -v
    echo "运行测试"
    mkdir -p "$TEST_OUTPUT"

#---------------------------------- 需要配置 -----------------------------------#
    key_output="$TEST_OUTPUT/output_mesh.glb"
    key_check_file=$key_output
    time "$PYTHON_BIN" "$BIN_PATH/exec" \
        "$CONFDATA_DIR/input_data" \
        "$CONFDATA_DIR/models" \
        "$key_output"
#------------------------------ END of 需要配置 --------------------------------#

    if test -e $key_check_file; then
        echo -e "\e[32m结果 $TEST_OUTPUT 存在，测试通过!\e[0m"
    else
        echo -e "\e[31m结果 $key_check_file 不存在，测试失败!\e[0m"
        exit 1
    fi

elif [ "$cmd" = "pack" ];then
    echo "Start packing"
    if [ ! -f "$BIN_PATH/exec" ];then
        echo -e "\e[31m二进制程序 \"$BIN_PATH/exec\"  不存在，请运行 \"$0 build\" 构建！ \e[0m"
        exit 1
    fi
    mkdir -p "$PACK_DIR"
    cd "$CONFDATA_DIR"
#---------------------------------- 需要配置 -----------------------------------#
    echo "test 压缩包获取"
    tar -zvcf test.tar.gz input_data
    mv test.tar.gz "$PACK_DIR"

    echo "data 压缩包获取"
    tar -zvcf data.tar.gz models
    mv data.tar.gz "$PACK_DIR"

    echo "runner 启动器脚本获取"
    cp "$SCRIPT_DIR/runner.sh" "$PACK_DIR/runner"
    chmod +x "$PACK_DIR/runner"
#------------------------------ END of 需要配置 --------------------------------#

    echo "获取 conda 环境配置"
    conda pack -n $CONDA_ENV_NAME -o "$PACK_DIR/conda_environment.tar.gz"
    cd "$PACK_DIR"
    tar -zvcf lib.tar.gz conda_environment.tar.gz
    rm -f "$PACK_DIR/conda_environment.tar.gz"

    echo "复制可执行程序"
    cp "$BIN_PATH/exec" "$PACK_DIR/exec"

    echo "Project info:"
    "${CONDA_PATH}/bin/python" "$PACK_DIR/exec" -v
    echo "Pack to:"
    echo "$PACK_DIR"
else
    echo "Usage: $0 clear | build | test | pack"
    exit 1
fi
