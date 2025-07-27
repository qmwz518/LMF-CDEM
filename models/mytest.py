try:
    import pywencai as wc
except ImportError:
    print("未找到pywencai库，请使用 pip install pywencai 进行安装")
    raise

def get_latest_consecutive_limit_up_stocks():
    """
    使用pywencai库查询最新连板股票
    
    Returns:
        DataFrame: 包含最新连板股票信息的DataFrame
    """
    # 使用问财查询最新连板股票
    query = "最新连板股票"
    result = wc.get(query)
    return result

if __name__ == "__main__":
    try:
        stocks = get_latest_consecutive_limit_up_stocks()
        print(stocks)
    except Exception as e:
        print(f"查询连板股票时出错: {e}")
