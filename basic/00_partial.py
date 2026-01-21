from functools import partial


def order_tea(tea_type, sugar, ice):
    print(f"下单：{tea_type}, {sugar}, {ice}")


# 使用 partial 固定部分参数
order_my_favorite = partial(order_tea, sugar="全糖", ice="去冰")

# 现在只需要传入剩下的那个参数
order_my_favorite("乌龙茶")
# 输出：下单：乌龙茶, 全糖, 去冰

order_my_favorite("茉莉绿茶")
# 输出：下单：茉莉绿茶, 全糖, 去冰
