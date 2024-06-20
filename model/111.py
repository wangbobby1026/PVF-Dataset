"""
@FileName：111.py\n
@Description：\n
@Author：WBobby\n
@Department：CUG\n
@Time：2024/6/17 22:40\n
"""


def my_decorator(func):
    def wrapper():
        print("装饰器添加的功能")
        func()
        func()
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")


say_hello()
