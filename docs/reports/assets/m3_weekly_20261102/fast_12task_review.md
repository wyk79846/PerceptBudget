# M3+Fast 新版12任务对照汇总

新版完整12任务/1次模型加载；历史M3、q30同任务seed7/init0。能量为完整任务含收尾的板级总能量；时间不含收尾。每任务每组一次。

|方法|成功|总能量J|总任务时间s|推理次数|
|---|---:|---:|---:|---:|
|q30|10/12|1859.42|126.11|79|
|m3|8/12|2187.45|152.58|109|
|m3_fast_v3|8/12|2054.79|161.22|97|

Fast共33/97次（34.0%），所有任务0断供，频率审核通过并恢复；最低可用RAM1121MiB。

|任务|方法|结果|步数|时间s|能量J|推理次数|
|---|---|---|---:|---:|---:|---:|
|libero_spatial 0|q30|success|76|5.769|79.816|3|
|libero_spatial 0|m3|success|77|6.110|81.347|4|
|libero_spatial 0|m3_fast_v3|success|77|6.202|70.798|3|
|libero_spatial 3|q30|success|84|6.356|86.170|3|
|libero_spatial 3|m3|success|88|6.894|85.927|4|
|libero_spatial 3|m3_fast_v3|success|86|6.813|84.956|4|
|libero_spatial 6|q30|success|109|7.941|109.804|4|
|libero_spatial 6|m3|success|112|8.628|109.967|5|
|libero_spatial 6|m3_fast_v3|success|113|8.785|98.668|4|
|libero_object 0|q30|success|134|7.583|112.380|5|
|libero_object 0|m3|success|147|8.674|123.168|6|
|libero_object 0|m3_fast_v3|success|143|8.473|97.076|4|
|libero_object 3|q30|success|126|7.403|109.517|5|
|libero_object 3|m3|timeout|280|15.076|230.830|12|
|libero_object 3|m3_fast_v3|timeout|280|15.099|189.402|9|
|libero_object 6|q30|success|147|8.244|120.918|5|
|libero_object 6|m3|success|149|8.472|119.218|6|
|libero_object 6|m3_fast_v3|success|154|8.875|108.765|5|
|libero_goal 0|q30|timeout|300|16.264|242.524|10|
|libero_goal 0|m3|timeout|300|16.959|215.067|9|
|libero_goal 0|m3_fast_v3|timeout|300|17.128|198.869|8|
|libero_goal 3|q30|success|161|8.957|134.186|6|
|libero_goal 3|m3|success|176|10.155|161.888|9|
|libero_goal 3|m3_fast_v3|success|185|10.591|128.394|6|
|libero_goal 6|q30|success|98|5.907|86.401|4|
|libero_goal 6|m3|timeout|300|17.324|264.162|14|
|libero_goal 6|m3_fast_v3|timeout|300|17.363|248.440|13|
|libero_10 0|q30|timeout|520|28.057|424.547|18|
|libero_10 0|m3|timeout|520|30.422|463.231|24|
|libero_10 0|m3_fast_v3|success|285|17.382|217.603|10|
|libero_10 3|q30|success|222|11.891|177.678|8|
|libero_10 3|m3|success|232|12.463|170.449|8|
|libero_10 3|m3_fast_v3|success|323|17.457|242.868|12|
|libero_10 6|q30|success|217|11.741|175.475|8|
|libero_10 6|m3|success|211|11.408|162.198|8|
|libero_10 6|m3_fast_v3|timeout|520|27.049|368.957|19|

## 解读

新版对M3总体能量−6.1%，成功率均8/12。共同成功的7个任务：能量−2.4%、时间+9.4%，尚无明显优势。相较q30少成功2个、总体能量+10.5%；共同成功7任务：能量+1.3%、时间+18.4%。

与M3相比，LIBERO10 task0由失败变成功，task6由成功变失败；总体能量差不能全部归因于Fast省电。

当前代码固定q5的Goal0额外复核成功123步。该对照取得全局GPU互斥锁后才创建模型worker，未与12任务GPU计算重叠。

常驻模型已完成12任务未触发内存保护，Fast确实执行。暂不替换M3基线；继续检查接触阶段与刷新时机。

原始批次：/home/wyk/workspace/任务产出/perceptbudget/M4/20261030T120850.170972Z-acceptance-m3_fast-d33224be
