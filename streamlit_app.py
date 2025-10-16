import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pyngrok import ngrok
import warnings
import threading

warnings.filterwarnings('ignore')

# 设置中文字体和样式
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'SimHei'  # 设置黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


class ConstructionCashFlowMC:
    def __init__(self, num_simulations=10000):
        self.num_simulations = num_simulations
        self.results = {}

    def generate_triangular_samples(self, low, mode, high, size):
        return np.random.triangular(low, mode, high, size)

    def generate_weather_impact(self, task_durations, weather_risk_level, season_sensitivity):
        """
        生成天气对工期的影响
        weather_risk_level: 天气风险等级 (1-5)
        season_sensitivity: 季节敏感性 (1-5)
        """
        weather_delays = {}

        # 不同任务的天气敏感性系数
        task_sensitivity = {
            'site_prep': 0.8,  # 场地准备对天气敏感
            'foundation': 0.6,  # 基础工程对天气敏感
            'structure': 0.4,  # 主体结构中等敏感
            'enclosure': 0.3,  # 围护结构较低敏感
            'mep': 0.1,  # 机电安装室内工作
            'finishing': 0.1  # 内部装修室内工作
        }

        for task, duration in task_durations.items():
            sensitivity = task_sensitivity[task]
            # 天气影响因子 = 基础风险 × 任务敏感性 × 季节敏感性
            weather_factor = (weather_risk_level / 5.0) * sensitivity * (season_sensitivity / 5.0)

            # 生成天气延迟：基于正态分布，均值为工期的weather_factor比例
            base_delay = duration * weather_factor * 0.1  # 基础延迟为工期的10% × weather_factor
            weather_delay = np.random.normal(base_delay, base_delay * 0.3)
            weather_delay = np.maximum(weather_delay, 0)  # 确保延迟不为负

            weather_delays[task] = weather_delay

        return weather_delays

    def run_simulation(self, task_params, approval_delay, payment_period, milestone_amounts,
                       weather_risk_level=3, season_sensitivity=3):
        # 模拟各任务工期
        task_durations = {}
        for task, params in task_params.items():
            task_durations[task] = self.generate_triangular_samples(
                params[0], params[1], params[2], self.num_simulations
            )

        # 模拟天气影响
        weather_delays = self.generate_weather_impact(task_durations, weather_risk_level, season_sensitivity)

        # 应用天气延迟
        for task in task_durations.keys():
            task_durations[task] += weather_delays[task]

        # 模拟审批延迟
        approval_delays = self.generate_triangular_samples(
            approval_delay[0], approval_delay[1], approval_delay[2], self.num_simulations
        )

        # 计算里程碑达成时间
        milestone1 = task_durations['site_prep'] + task_durations['foundation']
        milestone2 = milestone1 + task_durations['structure'] + task_durations['enclosure']
        milestone3 = milestone2 + task_durations['mep'] + task_durations['finishing']

        # 计算现金流入时间
        cash_flow1 = milestone1 + approval_delays + payment_period
        cash_flow2 = milestone2 + approval_delays + payment_period
        cash_flow3 = milestone3 + approval_delays + payment_period

        # 转换为月份
        cash_flow1_months = cash_flow1 / 30
        cash_flow2_months = cash_flow2 / 30
        cash_flow3_months = cash_flow3 / 30

        self.results = {
            'milestone1': cash_flow1_months,
            'milestone2': cash_flow2_months,
            'milestone3': cash_flow3_months,
            'task_durations': task_durations,
            'weather_delays': weather_delays,
            'milestone_amounts': milestone_amounts,
            'cash_flow_days': {
                'milestone1': cash_flow1,
                'milestone2': cash_flow2,
                'milestone3': cash_flow3
            }
        }

        return self.results

    def calculate_statistics(self):
        stats_dict = {}
        milestones = ['基础完成付款', '结构封顶付款', '项目竣工付款']
        data_keys = ['milestone1', 'milestone2', 'milestone3']

        for name, key in zip(milestones, data_keys):
            data = self.results[key]
            stats_dict[name] = {
                'P10': np.percentile(data, 10),
                'P50': np.percentile(data, 50),
                'P90': np.percentile(data, 90),
                'mean': np.mean(data),
                'std': np.std(data)
            }
        return stats_dict

    def plot_monte_carlo_paths(self, num_paths=50):
        """绘制蒙特卡罗模拟路径"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 生成时间序列的现金流路径
        milestones = ['milestone1', 'milestone2', 'milestone3']
        milestone_times = []
        milestone_values = []

        for i, milestone in enumerate(milestones):
            times = self.results[milestone]
            values = [self.results['milestone_amounts'][['基础完成', '结构封顶', '项目竣工'][i]]] * len(times)
            milestone_times.append(times)
            milestone_values.append(values)

        # 绘制随机路径样本
        colors = plt.cm.viridis(np.linspace(0, 1, num_paths))

        for i in range(min(num_paths, self.num_simulations)):
            path_times = []
            path_values = []

            for j, milestone in enumerate(milestones):
                time = self.results[milestone][i]
                value = self.results['milestone_amounts'][['基础完成', '结构封顶', '项目竣工'][j]]
                path_times.append(time)
                path_values.append(value)

            # 连接路径点
            ax.plot(path_times, path_values, 'o-', color=colors[i], alpha=0.3, linewidth=0.8)

        # 绘制平均路径
        avg_times = [np.mean(self.results[milestone]) for milestone in milestones]
        avg_values = [self.results['milestone_amounts'][name] for name in ['基础完成', '结构封顶', '项目竣工']]
        ax.plot(avg_times, avg_values, 'ro-', linewidth=3, markersize=8, label='平均路径')

        # 添加置信区间
        for i, milestone in enumerate(milestones):
            times = self.results[milestone]
            lower = np.percentile(times, 10)
            upper = np.percentile(times, 90)
            value = self.results['milestone_amounts'][['基础完成', '结构封顶', '项目竣工'][i]]
            ax.errorbar(np.mean(times), value, xerr=[[np.mean(times) - lower], [upper - np.mean(times)]],
                        fmt='none', ecolor='red', elinewidth=2, capsize=5, capthick=2)

        ax.set_xlabel('时间 (月)')
        ax.set_ylabel('现金流金额 (万元)')
        ax.set_title('蒙特卡罗模拟路径可视化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_task_sensitivity(self):
        """绘制任务工期敏感性分析"""
        tasks_chinese = ['场地准备', '基础工程', '主体结构', '围护结构', '机电安装', '内部装修']
        task_keys = ['site_prep', 'foundation', 'structure', 'enclosure', 'mep', 'finishing']

        # 计算各任务工期与总工期的相关性
        total_duration = self.results['milestone3']
        correlations = []

        for task_key in task_keys:
            correlation = np.corrcoef(self.results['task_durations'][task_key], total_duration)[0, 1]
            correlations.append(correlation)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 子图1: 任务工期与总工期相关性
        bars = ax1.barh(tasks_chinese, correlations, color=plt.cm.Set3(np.linspace(0, 1, len(tasks_chinese))))
        ax1.set_xlabel('与总工期的相关系数')
        ax1.set_title('任务工期敏感性分析')
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.8)

        # 在条形图上添加数值
        for bar, correlation in zip(bars, correlations):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{correlation:.3f}', ha='left', va='center')

        # 子图2: 天气影响分析
        weather_impact = []
        for task_key in task_keys:
            original_mean = np.mean(self.generate_triangular_samples(
                st.session_state.get(f'{task_key}_opt', 20),
                st.session_state.get(f'{task_key}_ml', 25),
                st.session_state.get(f'{task_key}_pes', 35),
                10000
            ))
            with_weather_mean = np.mean(self.results['task_durations'][task_key])
            impact = (with_weather_mean - original_mean) / original_mean * 100
            weather_impact.append(impact)

        bars2 = ax2.barh(tasks_chinese, weather_impact, color=plt.cm.Set3(np.linspace(0, 1, len(tasks_chinese))))
        ax2.set_xlabel('天气导致的工期增加 (%)')
        ax2.set_title('天气对各项任务的影响')
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.8)

        for bar, impact in zip(bars2, weather_impact):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{impact:.1f}%', ha='left', va='center')

        plt.tight_layout()
        return fig


def main():
    st.set_page_config(
        page_title="建筑项目现金流风险分析",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏗️ 建筑项目现金流蒙特卡罗模拟分析")
    st.markdown("---")

    # 侧边栏 - 参数输入
    st.sidebar.header("📊 模拟参数设置")

    num_simulations = st.sidebar.number_input(
        "模拟次数",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000
    )

    st.sidebar.subheader("任务工期参数 (天)")

    task_params = {}
    tasks = {
        'site_prep': '场地准备',
        'foundation': '基础工程',
        'structure': '主体结构',
        'enclosure': '围护结构',
        'mep': '机电安装',
        'finishing': '内部装修'
    }

    for key, name in tasks.items():
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            optimistic = st.number_input(f"{name}-乐观", value=20 if key == 'site_prep' else 45, key=f"{key}_opt")
        with col2:
            most_likely = st.number_input(f"{name}-可能", value=25 if key == 'site_prep' else 55, key=f"{key}_ml")
        with col3:
            pessimistic = st.number_input(f"{name}-悲观", value=35 if key == 'site_prep' else 75, key=f"{key}_pes")
        task_params[key] = (optimistic, most_likely, pessimistic)

    st.sidebar.subheader("环境风险参数")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        weather_risk_level = st.slider("天气风险等级", 1, 5, 3,
                                       help="1=气候稳定，5=气候多变多极端天气")
    with col2:
        season_sensitivity = st.slider("季节敏感性", 1, 5, 3,
                                       help="1=全年可施工，5=季节性施工影响大")

    st.sidebar.subheader("其他参数")

    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        approval_opt = st.number_input("审批延迟-乐观", value=10)
    with col2:
        approval_ml = st.number_input("审批延迟-可能", value=20)
    with col3:
        approval_pes = st.number_input("审批延迟-悲观", value=45)

    approval_delay = (approval_opt, approval_ml, approval_pes)
    payment_period = st.sidebar.number_input("付款账期 (天)", value=30)

    st.sidebar.subheader("里程碑付款金额 (万元)")
    milestone_amounts = {
        '基础完成': st.sidebar.number_input("基础完成金额", value=500.0),
        '结构封顶': st.sidebar.number_input("结构封顶金额", value=800.0),
        '项目竣工': st.sidebar.number_input("项目竣工金额", value=1200.0)
    }

    # 运行模拟按钮
    if st.sidebar.button("🚀 运行蒙特卡罗模拟", use_container_width=True):
        with st.spinner("正在进行蒙特卡罗模拟，请稍候..."):
            # 初始化模拟器
            mc_simulator = ConstructionCashFlowMC(num_simulations=num_simulations)

            # 运行模拟
            results = mc_simulator.run_simulation(
                task_params, approval_delay, payment_period, milestone_amounts,
                weather_risk_level, season_sensitivity
            )

            # 计算统计结果
            stats_dict = mc_simulator.calculate_statistics()

            # 显示结果
            st.success("模拟完成！")

            # 统计摘要
            st.header("📈 模拟结果摘要")

            col1, col2, col3 = st.columns(3)
            milestones = ['基础完成付款', '结构封顶付款', '项目竣工付款']
            amount_keys = ['基础完成', '结构封顶', '项目竣工']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

            for i, (milestone, amount_key) in enumerate(zip(milestones, amount_keys)):
                with col1 if i == 0 else col2 if i == 1 else col3:
                    stats = stats_dict[milestone]
                    amount = milestone_amounts[amount_key]

                    st.metric(
                        label=f"{milestone} ({amount}万元)",
                        value=f"{stats['P50']:.1f} 个月",
                        delta=f"±{stats['std']:.1f} 个月"
                    )

                    with st.expander("详细统计"):
                        st.write(f"**P10 (乐观):** {stats['P10']:.1f} 个月")
                        st.write(f"**P50 (最可能):** {stats['P50']:.1f} 个月")
                        st.write(f"**P90 (保守):** {stats['P90']:.1f} 个月")
                        st.write(f"**平均值:** {stats['mean']:.1f} 个月")
                        st.write(f"**标准差:** {stats['std']:.1f} 个月")

            # 环境风险影响分析
            st.header("🌦️ 环境风险影响分析")

            col1, col2 = st.columns(2)
            with col1:
                # 天气影响统计
                total_weather_delay = 0
                for task, delays in results['weather_delays'].items():
                    total_weather_delay += np.mean(delays)

                st.metric(
                    "平均天气延迟",
                    f"{total_weather_delay:.1f} 天",
                    delta=f"约 {total_weather_delay / 30:.1f} 个月"
                )

            with col2:
                # 天气对总工期的影响
                base_duration = sum([params[1] for params in task_params.values()])  # 最可能工期总和
                simulated_duration = np.mean(results['milestone3']) * 30  # 模拟平均总工期(天)
                weather_impact_pct = (simulated_duration - base_duration) / base_duration * 100

                st.metric(
                    "天气对总工期影响",
                    f"{weather_impact_pct:.1f}%",
                    delta=f"增加 {simulated_duration - base_duration:.0f} 天"
                )

            # 可视化图表
            st.header("📊 可视化分析")

            # 图表1: 里程碑时间分布
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            data_to_plot = [results['milestone1'], results['milestone2'], results['milestone3']]
            box_plot = ax1.boxplot(data_to_plot, labels=milestones, patch_artist=True)

            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)

            ax1.set_title('里程碑付款时间分布 (含天气影响)')
            ax1.set_ylabel('月份')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

            # 图表2: 概率密度分布
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            for i, (milestone, color) in enumerate(zip(milestones, colors)):
                data = data_to_plot[i]
                sns.kdeplot(data, ax=ax2, label=milestone, color=color, linewidth=2)

                p50 = stats_dict[milestone]['P50']
                ax2.axvline(p50, color=color, linestyle='--', alpha=0.7)

            ax2.set_title('现金流时间概率密度分布')
            ax2.set_xlabel('月份')
            ax2.set_ylabel('概率密度')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

            # 新增图表3: 蒙特卡罗模拟路径
            st.subheader("🔄 蒙特卡罗模拟路径")
            fig3 = mc_simulator.plot_monte_carlo_paths(num_paths=50)
            st.pyplot(fig3)

            # 新增图表4: 任务敏感性分析
            st.subheader("📋 任务工期敏感性分析")
            fig4 = mc_simulator.plot_task_sensitivity()
            st.pyplot(fig4)

            # 图表5: 现金流时间线风险
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            target_months = [3, 6, 9, 12, 15, 18, 21, 24]
            prob_data = []

            for month in target_months:
                row = [month]
                for key in ['milestone1', 'milestone2', 'milestone3']:
                    prob = np.mean(results[key] <= month) * 100
                    row.append(prob)
                prob_data.append(row)

            prob_df = pd.DataFrame(prob_data, columns=['月份', '基础完成', '结构封顶', '项目竣工'])

            x = range(len(target_months))
            width = 0.25

            ax5.bar([i - width for i in x], prob_df['基础完成'], width,
                    label='基础完成', color=colors[0], alpha=0.8)
            ax5.bar(x, prob_df['结构封顶'], width,
                    label='结构封顶', color=colors[1], alpha=0.8)
            ax5.bar([i + width for i in x], prob_df['项目竣工'], width,
                    label='项目竣工', color=colors[2], alpha=0.8)

            ax5.set_title('各时间点前收到付款的概率')
            ax5.set_xlabel('月份')
            ax5.set_ylabel('概率 (%)')
            ax5.set_xticks(x)
            ax5.set_xticklabels(target_months)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            st.pyplot(fig5)

            # 现金流风险分析
            st.header("⚠️ 现金流风险分析")

            target_months_analysis = [3, 6, 9, 12, 18, 24]
            data_keys = ['milestone1', 'milestone2', 'milestone3']

            risk_data = []
            for month in target_months_analysis:
                row = {'月份': month}
                total_expected = 0
                for i, (key, amount_key) in enumerate(zip(data_keys, amount_keys)):
                    prob = np.mean(results[key] <= month) * 100
                    expected = prob / 100 * milestone_amounts[amount_key]
                    total_expected += expected
                    row[milestones[i]] = f"{prob:.1f}%"
                    row[f"{milestones[i]}_金额"] = f"{expected:.1f}万元"
                row['总预期现金流'] = f"{total_expected:.1f}万元"
                risk_data.append(row)

            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True)

            # 现金流覆盖率警告
            six_month_cash = sum(np.mean(results[key] <= 6) * milestone_amounts[amount_key]
                                 for key, amount_key in zip(data_keys, amount_keys))
            twelve_month_cash = sum(np.mean(results[key] <= 12) * milestone_amounts[amount_key]
                                    for key, amount_key in zip(data_keys, amount_keys))
            total_contract = sum(milestone_amounts.values())

            col1, col2 = st.columns(2)
            with col1:
                coverage_6m = six_month_cash / total_contract * 100
                st.metric("6个月现金流覆盖率", f"{coverage_6m:.1f}%")
                if coverage_6m < 30:
                    st.error("⚠️ 前6个月现金流覆盖率较低，可能存在资金压力")

            with col2:
                coverage_12m = twelve_month_cash / total_contract * 100
                st.metric("12个月现金流覆盖率", f"{coverage_12m:.1f}%")
                if coverage_12m < 60:
                    st.warning("⚠️ 前12个月现金流覆盖率较低，需要关注资金安排")

    # 使用说明
    with st.sidebar.expander("💡 使用说明"):
        st.write("""
        1. **设置参数**: 在左侧输入项目各任务的工期估计
        2. **环境风险**: 调整天气风险等级和季节敏感性
        3. **运行模拟**: 点击运行按钮开始蒙特卡罗模拟
        4. **查看结果**: 分析模拟结果和风险提示

        **新增功能**:
        - 🌦️ 天气影响模拟
        - 🔄 蒙特卡罗路径可视化  
        - 📋 任务敏感性分析
        """)


if __name__ == "__main__":
    main()