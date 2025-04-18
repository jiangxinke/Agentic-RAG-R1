import requests
import sys
import time
import threading

from src.data.prompt import build_system_tools

# 服务地址
url = "http://0.0.0.0:12333/chat/"
system_prompt = build_system_tools()


def animate(stop_event):
    dots = [".", "..", "...", "....", "....."]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write("\r正在处理" + dots[i % 5])
        sys.stdout.flush()
        time.sleep(0.5)
        i += 1

    sys.stdout.write("\r" + " " * 20 + "\r")
    sys.stdout.flush()


print("xiaobei-r1 小北医生")

while True:
    user_input = input("用户输入: ")
    if user_input.lower() == "exit":
        break

    query = {"text": f"{system_prompt} {user_input}"}

    stop_event = threading.Event()
    animation_thread = threading.Thread(target=animate, args=(stop_event,), daemon=True)
    animation_thread.start()

    try:
        response = requests.post(url, json=query)

        stop_event.set()
        animation_thread.join()

        if response.status_code == 200:
            result = response.json()
            print("BOT:", result.get("result", ""))
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        stop_event.set()
        animation_thread.join()
        print(f"请求出错: {e}")
