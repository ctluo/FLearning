import XLSX
using Base: run
using Dates
using SymbolicUtils
using Statistics
using Base64

nthread = string(Sys.CPU_THREADS) # 获取当前电脑CPU的线程数

if get(ENV, "JULIA_NUM_THREADS", nothing) === nothing
    ENV["JULIA_NUM_THREADS"] = nthread # "8"  # 设置默认线程数
end

baseName = "Steady_CN_0-90"

nround=10 # Rounds to run: [1,100] 
maxNodes =20 # integer > 20
popsize= 50 # integer > 20
maxIter=200 # integer > 100
report_path = joinpath(@__DIR__, "Report.txt")
bestFormula_path = joinpath(@__DIR__, "bestFormula.txt")


# 从文件中读取数据集
datafileName = "./datasets/" * baseName *".xlsx"
xf = XLSX.readxlsx(datafileName )
# Close the file when done
# close(xf)
sheetnames=XLSX.sheetnames(xf)
# println(sheetnames)
sheet1 = xf[sheetnames[1]] # get a reference to a Worksheet
# println(sheet1)
rawdata=sheet1[:]
#  
NTrain=65; # for  "Steady_CN_0-90" and Steady_mZ0_0-90.xlsx
# NTrain=35; # for  "Steady_CN_90-180" and Steady_mZ0_90-180.xlsx

X_train=Float64.(rawdata[2:NTrain+1 1:end-2])
y_train=Float64.(rawdata[2:NTrain+1, end-1])

X_test=Float64.(rawdata[NTrain+2:end, 1:end-2])
y_test=Float64.(rawdata[NTrain+2:end, end-1])

for i in 1:nround
    println("第 $i 轮学习开始")	
	# Return ParetoF and update report_path and bestFormula_path
	ParetoF=FLearning.FLearning(X_train, y_train, X_test, y_test, popsize, maxNodes, maxIter, report_path, bestFormula_path)
	
    strR2=string(1.0-ParetoF[end].loss) # ParetoF[end].loss: 1-R2
    time_string=Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    file_list=[datafileName, report_path, bestFormula_path]
    run(`7z a ./Backups/$baseName.$strR2.$time_string.7z  $file_list`)


end # for i in 1:nround
