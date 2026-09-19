#!/usr/bin/env python3
from __future__ import annotations
from datetime import date
from market_data_sina import is_main_board
from trading_calendar import next_trading_day, previous_trading_day

def main():
    assert is_main_board("600000","浦发银行")
    assert is_main_board("000001","平安银行")
    for name in ("ST测试","*ST测试","S*ST测试","SST测试"):
        assert not is_main_board("600000",name), name
    assert not is_main_board("300001","测试")
    assert not is_main_board("688001","测试")
    assert not is_main_board("830001","测试")

    assert next_trading_day(date(2026,9,18)) == date(2026,9,21)
    assert next_trading_day(date(2026,9,24)) == date(2026,9,28)
    assert next_trading_day(date(2026,9,30)) == date(2026,10,8)
    assert previous_trading_day(date(2026,9,28)) == date(2026,9,24)
    print("one-to-two smoke tests: OK")

if __name__ == "__main__":
    main()
