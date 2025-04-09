from data.data_fetcher import get_forex_fetcher

def main():
    # データ取得クラスのインスタンスを取得
    fetcher = get_forex_fetcher()
    
    # 1年間の分析を実行
    analysis = fetcher.fetch_long_term_analysis('USD/JPY', years=1)
    
    # 結果の表示
    print('分析結果:')
    print(f'期間: {analysis.get("period")}')
    print(f'開始日: {analysis.get("start_date")}')
    print(f'終了日: {analysis.get("end_date")}')
    print(f'データ数: {analysis.get("total_days")}件')
    print(f'平均レート: {analysis.get("average_close"):.2f}')
    print(f'最高値: {analysis.get("max_close"):.2f}')
    print(f'最安値: {analysis.get("min_close"):.2f}')
    print(f'ボラティリティ: {analysis.get("volatility"):.2f}')
    print(f'シャープレシオ: {analysis.get("sharpe_ratio"):.2f}')
    
    # 月次リターンの統計
    monthly_returns = analysis.get("monthly_returns")
    if monthly_returns is not None:
        print('\n月次リターン統計:')
        print(f'平均: {monthly_returns.mean():.2%}')
        print(f'最大: {monthly_returns.max():.2%}')
        print(f'最小: {monthly_returns.min():.2%}')

if __name__ == '__main__':
    main() 