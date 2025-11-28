#!/usr/bin/env python3
# scripts/test_monitor.py
"""
測試監控和報告工具
"""

import time
import json
import subprocess
import psutil
from pathlib import Path
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class TestMonitor:
    def __init__(self, config_file="test_monitor_config.json"):
        self.config = self.load_config(config_file)
        self.results_history = []

    def load_config(self, config_file):
        """載入監控配置"""
        default_config = {
            "monitoring": {
                "interval_minutes": 60,
                "max_history": 100,
                "alert_thresholds": {
                    "failure_rate": 0.1,
                    "response_time": 5.0,
                    "memory_growth": 100,  # MB
                },
            },
            "notifications": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": [],
                },
                "slack": {"enabled": False, "webhook_url": ""},
            },
            "test_suites": [
                {
                    "name": "smoke",
                    "command": ["python", "tests/run_tests.py", "--smoke"],
                },
                {
                    "name": "unit",
                    "command": [
                        "python",
                        "-m",
                        "pytest",
                        "tests/test_core_modules.py",
                        "-q",
                    ],
                },
                {
                    "name": "integration",
                    "command": [
                        "python",
                        "-m",
                        "pytest",
                        "tests/test_api_endpoints.py",
                        "-q",
                    ],
                },
            ],
        }

        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            # 創建默認配置
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=2)
            return default_config

    def run_test_suite(self, test_config):
        """執行測試套件"""
        start_time = time.time()

        try:
            result = subprocess.run(
                test_config["command"],
                capture_output=True,
                text=True,
                timeout=300,  # 5分鐘超時
            )

            duration = time.time() - start_time

            return {
                "name": test_config["name"],
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat(),
            }

        except subprocess.TimeoutExpired:
            return {
                "name": test_config["name"],
                "success": False,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": "Test suite timed out",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "name": test_config["name"],
                "success": False,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def collect_system_metrics(self):
        """收集系統指標"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available": psutil.virtual_memory().available / 1024 / 1024,  # MB
            "disk_percent": psutil.disk_usage("/").percent,
            "timestamp": datetime.now().isoformat(),
        }

    def run_monitoring_cycle(self):
        """執行一個監控週期"""
        cycle_start = datetime.now()
        print(f"🔄 開始監控週期: {cycle_start}")

        # 收集系統指標
        system_metrics = self.collect_system_metrics()

        # 執行測試套件
        test_results = []
        for test_config in self.config["test_suites"]:
            print(f"   🧪 執行測試套件: {test_config['name']}")
            result = self.run_test_suite(test_config)
            test_results.append(result)

            status = "✅" if result["success"] else "❌"
            print(f"      {status} {result['duration']:.1f}s")

        # 創建監控結果
        monitoring_result = {
            "cycle_start": cycle_start.isoformat(),
            "cycle_end": datetime.now().isoformat(),
            "system_metrics": system_metrics,
            "test_results": test_results,
            "summary": {
                "total_tests": len(test_results),
                "passed_tests": sum(1 for r in test_results if r["success"]),
                "failed_tests": sum(1 for r in test_results if not r["success"]),
                "average_duration": (
                    sum(r["duration"] for r in test_results) / len(test_results)
                    if test_results
                    else 0
                ),
            },
        }

        # 保存到歷史記錄
        self.results_history.append(monitoring_result)

        # 保持歷史記錄大小限制
        max_history = self.config["monitoring"]["max_history"]
        if len(self.results_history) > max_history:
            self.results_history = self.results_history[-max_history:]

        # 檢查警報條件
        self.check_alerts(monitoring_result)

        # 保存結果
        self.save_results()

        return monitoring_result

    def check_alerts(self, result):
        """檢查警報條件"""
        thresholds = self.config["monitoring"]["alert_thresholds"]
        alerts = []

        # 檢查失敗率
        if result["summary"]["total_tests"] > 0:
            failure_rate = (
                result["summary"]["failed_tests"] / result["summary"]["total_tests"]
            )
            if failure_rate > thresholds["failure_rate"]:
                alerts.append(f"測試失敗率過高: {failure_rate:.1%}")

        # 檢查響應時間
        avg_duration = result["summary"]["average_duration"]
        if avg_duration > thresholds["response_time"]:
            alerts.append(f"平均響應時間過長: {avg_duration:.1f}s")

        # 檢查記憶體使用
        memory_percent = result["system_metrics"]["memory_percent"]
        if memory_percent > 90:
            alerts.append(f"記憶體使用率過高: {memory_percent:.1f}%")

        # 發送警報
        if alerts:
            self.send_alerts(alerts, result)

    def send_alerts(self, alerts, result):
        """發送警報通知"""
        alert_message = (
            f"""
🚨 Multi-Modal Lab 測試警報

時間: {result['cycle_start']}

警報內容:
"""
            + "\n".join(f"- {alert}" for alert in alerts)
            + f"""

系統狀態:
- CPU使用率: {result['system_metrics']['cpu_percent']:.1f}%
- 記憶體使用率: {result['system_metrics']['memory_percent']:.1f}%
- 測試通過率: {result['summary']['passed_tests']}/{result['summary']['total_tests']}

請檢查系統狀態並採取相應措施。
        """
        )

        print("🚨 警報觸發:")
        for alert in alerts:
            print(f"   - {alert}")

        # Email 通知
        if self.config["notifications"]["email"]["enabled"]:
            self.send_email_alert(alert_message)

        # Slack 通知 (如果配置了)
        if self.config["notifications"]["slack"]["enabled"]:
            self.send_slack_alert(alert_message)

    def send_email_alert(self, message):
        """發送郵件警報"""
        try:
            email_config = self.config["notifications"]["email"]

            msg = MimeMultipart()
            msg["From"] = email_config["username"]
            msg["Subject"] = "Multi-Modal Lab 測試警報"

            msg.attach(MimeText(message, "plain"))

            server = smtplib.SMTP(
                email_config["smtp_server"], email_config["smtp_port"]
            )
            server.starttls()
            server.login(email_config["username"], email_config["password"])

            for recipient in email_config["recipients"]:
                msg["To"] = recipient
                server.sendmail(email_config["username"], recipient, msg.as_string())

            server.quit()
            print("   📧 郵件警報已發送")

        except Exception as e:
            print(f"   ❌ 郵件發送失敗: {e}")

    def save_results(self):
        """保存監控結果"""
        results_file = Path("test_results/monitoring_history.json")
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(self.results_history, f, indent=2)

    def generate_report(self, days=7):
        """生成監控報告"""
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_results = [
            r
            for r in self.results_history
            if datetime.fromisoformat(r["cycle_start"]) > cutoff_date
        ]

        if not recent_results:
            return "沒有足夠的監控資料"

        # 計算統計資料
        total_cycles = len(recent_results)
        total_tests = sum(r["summary"]["total_tests"] for r in recent_results)
        total_passed = sum(r["summary"]["passed_tests"] for r in recent_results)

        avg_cpu = (
            sum(r["system_metrics"]["cpu_percent"] for r in recent_results)
            / total_cycles
        )
        avg_memory = (
            sum(r["system_metrics"]["memory_percent"] for r in recent_results)
            / total_cycles
        )

        report = f"""
📊 Multi-Modal Lab 監控報告 (過去 {days} 天)

監控摘要:
- 監控週期: {total_cycles}
- 總測試數: {total_tests}
- 測試通過率: {total_passed/total_tests:.1%}
- 平均CPU使用率: {avg_cpu:.1f}%
- 平均記憶體使用率: {avg_memory:.1f}%

最近警報:
"""

        # 添加最近的警報資訊
        alert_count = 0
        for result in recent_results[-10:]:  # 最近10個週期
            if result["summary"]["failed_tests"] > 0:
                alert_count += 1
                report += f"- {result['cycle_start']}: {result['summary']['failed_tests']} 個測試失敗\n"

        if alert_count == 0:
            report += "- 無警報記錄\n"

        return report

    def start_monitoring(self):
        """啟動持續監控"""
        interval = self.config["monitoring"]["interval_minutes"] * 60

        print(f"🚀 啟動 Multi-Modal Lab 測試監控")
        print(f"   監控間隔: {self.config['monitoring']['interval_minutes']} 分鐘")
        print(f"   測試套件: {[t['name'] for t in self.config['test_suites']]}")

        try:
            while True:
                cycle_result = self.run_monitoring_cycle()

                print(
                    f"✅ 監控週期完成 - 通過率: {cycle_result['summary']['passed_tests']}/{cycle_result['summary']['total_tests']}"
                )

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n⏹️  監控已停止")

            # 生成最終報告
            report = self.generate_report()
            print("\n" + report)

            return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Modal Lab Test Monitor")
    parser.add_argument(
        "--config", default="test_monitor_config.json", help="Configuration file"
    )
    parser.add_argument(
        "--once", action="store_true", help="Run monitoring once and exit"
    )
    parser.add_argument("--report", type=int, help="Generate report for last N days")

    args = parser.parse_args()

    monitor = TestMonitor(args.config)

    if args.report:
        report = monitor.generate_report(args.report)
        print(report)
        return 0
    elif args.once:
        result = monitor.run_monitoring_cycle()
        print(
            f"監控完成 - 通過率: {result['summary']['passed_tests']}/{result['summary']['total_tests']}"
        )
        return 0
    else:
        return monitor.start_monitoring()


if __name__ == "__main__":
    main()
