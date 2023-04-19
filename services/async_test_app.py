# --coding:utf-8--
"""
测试app的异步访问的能力，可以达到 55 it/s
"""
import asyncio
import time

import aiohttp
import json
import requests
from tqdm import tqdm

num_times = 500
url = "http://localhost:8000/predict"

data = {
    "goods_a": "商品描述A",
    "goods_b": "商品描述B"
}


async def async_request(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            response_text = await resp.text()
            return json.loads(response_text)


async def main():
    tasks = []
    for _ in tqdm(range(num_times)):
        task = asyncio.ensure_future(async_request(url, data))
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    return responses


if __name__ == "__main__":
    time_bigin = time.time()
    loop = asyncio.get_event_loop()
    responses = loop.run_until_complete(main())
    print(responses)
    time_end = time.time()
    print(f"{(num_times)/(time_end-time_bigin)} it/s")
