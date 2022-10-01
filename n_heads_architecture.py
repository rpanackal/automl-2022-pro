import torch.nn as nn
import torch


class SimpleBlock(nn.Module):
    def __init__(self, input_neurons, intermediate_neurons):
        super(SimpleBlock, self).__init__()

        self.lin = nn.Linear(input_neurons, intermediate_neurons)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(intermediate_neurons)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.lin(x)
        x = self.activation(x)
        x = self.bn(x)
        x = self.drop(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_neurons, intermediate_neurons, inter_layer_num):
        super(Classifier, self).__init__()
        self.inter_layer_num = inter_layer_num
        self.in_block = SimpleBlock(input_neurons, intermediate_neurons)

        for i in range(self.inter_layer_num):
            block_name = "block_"+str(i)
            block = SimpleBlock(intermediate_neurons, intermediate_neurons)
            self.add_module(block_name, block)

        self.lin_fin = nn.Linear(intermediate_neurons, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.in_block(x)
        for i in range(self.inter_layer_num):
            inter_block = getattr(self, "block_"+str(i))
            x = inter_block(x)
        x = self.lin_fin(x)
        x = self.sigmoid(x)
        return x


class NHeadsArchitecture(torch.nn.Module):
    def __init__(self, input_neurons, class_num, backbone_depth, classifiers_depth, intermediate_neurons):
        super(NHeadsArchitecture, self).__init__()
        self.backbone_depth = backbone_depth
        self.class_num = class_num

        self.in_block = SimpleBlock(input_neurons, intermediate_neurons)

        for i in range(self.backbone_depth):
            block_name = "block_"+str(i)
            block = SimpleBlock(intermediate_neurons, intermediate_neurons)
            self.add_module(block_name, block)

        for i in range(self.class_num):
            classifier_name = "classifier_"+str(i)
            classifier = Classifier(intermediate_neurons, intermediate_neurons, classifiers_depth)
            self.add_module(classifier_name, classifier)

    def forward(self, x):
        x = self.in_block(x)
        for i in range(self.backbone_depth):
            block = getattr(self, "block_"+str(i))
            x = block(x)

        prediction_list = []
        for i in range(self.class_num):
            classifier = getattr(self, "classifier_"+str(i))
            single_pred = classifier(x)
            prediction_list.append(single_pred)

        prediction = torch.cat(prediction_list, dim=1)
        return prediction
