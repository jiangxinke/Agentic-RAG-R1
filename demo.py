import requests
import sys
import time
import threading
from data.prompt import build_system_tools

# 服务地址
url = "http://0.0.0.0:12333/chat/"
system_prompt = build_system_tools()


def animate(stop_event):
    """用于在等待服务器响应时显示“正在处理...”的动画。"""
    dots = [".", "..", "...", "....", "....."]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write("\r正在处理" + dots[i % 5])
        sys.stdout.flush()
        time.sleep(0.5)
        i += 1

    # 动画停止后，清除提示行
    sys.stdout.write("\r" + " " * 20 + "\r")
    sys.stdout.flush()


print("xiaobei-r1 小北医生")

while True:
    user_input = input("用户输入: ")
    if user_input.lower() == "exit":
        break

    query = {"text": f"{system_prompt} {user_input}"}

    # 创建一个 Event，用于通知动画线程结束
    stop_event = threading.Event()
    animation_thread = threading.Thread(target=animate, args=(stop_event,), daemon=True)
    animation_thread.start()

    try:
        # 在这里发送请求
        response = requests.post(url, json=query)

        # 请求完成，停止动画
        stop_event.set()
        animation_thread.join()

        # 根据响应状态码做处理
        if response.status_code == 200:
            result = response.json()
            print("BOT:", result.get("result", ""))
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        # 出现异常也要停止动画
        stop_event.set()
        animation_thread.join()
        print(f"请求出错: {e}")
