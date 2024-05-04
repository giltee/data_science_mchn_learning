from pyspark import SparkContext
sc = SparkContext()

## Important terms
# - RRD: Resilitent Distributed Dataset
# - Transformation: Spark opteration that produces an RRD
# - Action: spark operation that produces a local object
# - Spark Job: Sequence of transformations on data with a final action

## Creating an RRD
arr = [1,2,3,4,5,6,7,8,9]
para = sc.parallelize(arr)
text_file = sc.textFile('./example.txt')

# print(para, text_file)


## Transformations
# filter
# map
# flatmap 
# sample
# union 
# distinct 
# sortBy
filt = para.filter(lambda no: no < 5)

# print(filt.collect())

## Actions
# collect(): converts RRD to in-memory list
# take(3): first 3 elements of RRD
# top(3) Top 3 elements of rrd
# takeSample(withReplacement=Boolean, 3): create a sample of 3 elements with replacement
# sum()
# mean()
# stdev()

words = text_file.map(lambda line: line.split())
# print(words.collect())

flat = text_file.flatMap(lambda line: line.split()).collect()
# print(flat)

# import services.txt
services = sc.textFile('./services.txt')
# print(services.collect())
services.map(lambda line: line.split()).take(3)

clean = services.map(lambda line: line[1:] if line[0] == '#' else line)
clean = clean.map(lambda line: line.split())

# print(clean.collect())

state_amount = clean.map(lambda col: (col[3], col[-1]))
reduct = state_amount.reduceByKey(lambda amt1, amt2: float(amt1) + float(amt2))

filt = reduct.filter(lambda x: not x[0] == 'State')
# print(filt.collect())

final_step = filt.sortBy(lambda amount: amount[1], ascending=False)
print(final_step.collect())