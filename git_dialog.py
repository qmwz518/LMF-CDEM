#!/usr/bin/env python3
"""
Interactive Git Operations Script
通过对话式菜单执行常用Git操作
"""

import os
import subprocess


def run_cmd(cmd, cwd=None):
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"命令执行出错: {e}")
        return False


def main():
    print("欢迎使用对话式Git操作脚本！\n")
    repo_path = input("请输入Git仓库路径（直接回车为当前目录）: ").strip() or os.getcwd()
    print(f"当前仓库路径: {repo_path}\n")

    while True:
        print("请选择要执行的操作：")
        print("1. 初始化仓库 (git init)")
        print("2. 添加文件 (git add)")
        print("3. 提交更改 (git commit)")
        print("4. 推送到远程 (git push)")
        print("5. 拉取远程 (git pull)")
        print("6. 查看状态 (git status)")
        print("7. 查看日志 (git log)")
        print("8. 分支管理 (git branch)")
        print("9. 远程管理 (git remote)")
        print("0. 退出")
        choice = input("请输入操作编号: ").strip()

        if choice == "1":
            run_cmd(["git", "init"], cwd=repo_path)
        elif choice == "2":
            files = input("请输入要添加的文件（用空格分隔，留空为全部）: ").strip()
            if files:
                run_cmd(["git", "add"] + files.split(), cwd=repo_path)
            else:
                run_cmd(["git", "add", "."], cwd=repo_path)
        elif choice == "3":
            msg = input("请输入提交信息: ").strip()
            if msg:
                run_cmd(["git", "commit", "-m", msg], cwd=repo_path)
            else:
                print("提交信息不能为空！")
        elif choice == "4":
            remote = input("远程名（默认origin，回车跳过）: ").strip() or "origin"
            branch = input("分支名（留空为当前分支）: ").strip()
            cmd = ["git", "push", remote]
            if branch:
                cmd.append(branch)
            run_cmd(cmd, cwd=repo_path)
        elif choice == "5":
            remote = input("远程名（默认origin，回车跳过）: ").strip() or "origin"
            branch = input("分支名（留空为当前分支）: ").strip()
            cmd = ["git", "pull", remote]
            if branch:
                cmd.append(branch)
            run_cmd(cmd, cwd=repo_path)
        elif choice == "6":
            run_cmd(["git", "status"], cwd=repo_path)
        elif choice == "7":
            num = input("显示最近几条提交（默认10）: ").strip()
            num = num if num.isdigit() else "10"
            run_cmd(["git", "log", f"-{num}", "--oneline"], cwd=repo_path)
        elif choice == "8":
            print("a) 查看所有分支\nb) 创建新分支\nc) 切换分支")
            b_choice = input("请选择分支操作: ").strip().lower()
            if b_choice == "a":
                run_cmd(["git", "branch", "-a"], cwd=repo_path)
            elif b_choice == "b":
                name = input("新分支名: ").strip()
                if name:
                    run_cmd(["git", "checkout", "-b", name], cwd=repo_path)
            elif b_choice == "c":
                name = input("要切换到的分支名: ").strip()
                if name:
                    run_cmd(["git", "checkout", name], cwd=repo_path)
        elif choice == "9":
            print("a) 查看远程\nb) 添加远程\nc) 删除远程")
            r_choice = input("请选择远程操作: ").strip().lower()
            if r_choice == "a":
                run_cmd(["git", "remote", "-v"], cwd=repo_path)
            elif r_choice == "b":
                name = input("远程名: ").strip()
                url = input("远程URL: ").strip()
                if name and url:
                    run_cmd(["git", "remote", "add", name, url], cwd=repo_path)
            elif r_choice == "c":
                name = input("要删除的远程名: ").strip()
                if name:
                    run_cmd(["git", "remote", "remove", name], cwd=repo_path)
        elif choice == "0":
            print("再见！")
            break
        else:
            print("无效的选择，请重新输入。\n")


if __name__ == "__main__":
    main() 