# 绘制一张4个子图的图像，存为.png格式
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()  # 展平为一维数组便于索引
i = 0

# 创建线条对象存储列表
lines = []
labels = []

### 绘制效率图
for p in problem:
    for n in noise:
        if n == 'OE':
            x = quality_data['error_prop_list']
            y_scale = [0, 31]
        else:
            x = quality_data['prop_prob_list']
            y_scale = [20, 51]
        ax = axes[i]
        i += 1

        for m in methods:
            # 对于每一行，计算SL_time和MLSL_time的差值除以SL_time，得到求解时间降低百分比
            time_improvement = (time_data['SL'+'_'+p+'_'+m+'_'+n] - time_data['MLSL'+'_'+p+'_'+m+'_'+n]) / time_data['SL'+'_'+p+'_'+m+'_'+n] * 100
            # 对于每一行，计算SL_quality和MLSL_quality的差值除以SL_quality，得到求解质量降低百分比
            quality_improvement = (quality_data['SL'+'_'+p+'_'+m+'_'+n] - quality_data['MLSL'+'_'+p+'_'+m+'_'+n]) / quality_data['SL'+'_'+p+'_'+m+'_'+n] * 100
            # 计算商
            y = time_improvement / quality_improvement
            x_plot = x[y_scale[0]:y_scale[1]]
            y_plot = y[y_scale[0]:y_scale[1]]
            
            # 处理无穷大值
            y_plot = y_plot.replace([np.inf, -np.inf], [1, -1])
            # 将高于100和低于-100的节点值替换为1和-1
            y_plot = y_plot.apply(lambda v: 1 if v > 100 else (-1 if v < -100 else v))
            
            # 绘制曲线并保存线条对象
            line, = ax.plot(x_plot, y_plot, marker="o", zorder=1, label=m)
            if i == 1:  # 只在第一次循环时保存图例标签
                lines.append(line)
                labels.append(m)
                
            # 筛选出y_plot中的1和-1的索引
            good_indices = np.where(y_plot == 1)[0].tolist()
            if n == 'PP':
                good_indices = [i+20 for i in good_indices]
            x_good = x_plot[good_indices]
            y_good = y_plot[good_indices]
            bad_indices =  np.where(y_plot == -1)[0].tolist()
            if n == 'PP':
                bad_indices = [i+20 for i in bad_indices]
            x_bad = x_plot[bad_indices]
            y_bad = y_plot[bad_indices]
            
            # good点使用绿色圆点，bad点使用红色圆点标记为散点
            ax.scatter(x_good, y_good, c='g', marker="x", zorder=2)
            ax.scatter(x_bad, y_bad, c='r', marker="x", zorder=3)
            # 加上曲线Y=1，使用红色
            ax.axhline(y=1, color='r', linestyle='--', label='y=1')
            if n == 'OE':
                ax.set_xlabel("Pe")
            else:
                ax.set_xlabel("Pr")
            ax.set_ylabel("Improvement Ratio")
            ax.set_title(p+'_'+n)

# 添加统一图例
fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 0.9))

plt.tight_layout() 
plt.show()