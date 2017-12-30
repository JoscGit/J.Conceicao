from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

# cria-se um conjunto de entradas (dataset) para treinamento
# são passadas as dimensões dos vetores de entrada e do objetivo
dataset = SupervisedDataSet(2,1)

# adiciona-se as amostras (XOR)
dataset.addSample([1,1], [0])
dataset.addSample([1,0], [1])
dataset.addSample([0,1], [1])
dataset.addSample([0,0], [0])

# construindo a rede do tipo FeedForward
network = buildNetwork(dataset.indim, 2, dataset.outdim, bias=True)

# o procedimento utilizado para treinar a rede será o backpropagation
trainer = BackpropTrainer(network, dataset, learningrate=0.01, momentum=0.99)

# treinando a rede
for epoch in range(1000):
	trainer.train()

# teste da rede
test_data = SupervisedDataSet(2,1)
test_data.addSample([1,1], [0])
test_data.addSample([1,0], [1])
test_data.addSample([0,1], [1])
test_data.addSample([0,0], [0])

trainer.testOnData(test_data, verbose=True)