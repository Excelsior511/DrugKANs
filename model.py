import torch
from kan import KANLinear


class my_model(torch.nn.Module):
    def __init__(self, drug_dim, protein_dim):
        super(my_model, self).__init__()
        self.input_dim = 32
        self.hid_dim = 64
        self.output_dim = 32
        self.input_drug = torch.nn.Linear(drug_dim, self.input_dim)
        self.input_protein = torch.nn.Linear(protein_dim, self.input_dim)
        self.SE_layer1 = Light_SE()
        self.SE_layer2 = Light_SE()

        self.drug_fc1 = KANLinear(self.input_dim, self.hid_dim)
        self.drug_fc2 = KANLinear(self.hid_dim, self.hid_dim)
        self.drug_fc4 = KANLinear(self.hid_dim, self.hid_dim)
        self.drug_fc3 = KANLinear(self.hid_dim, self.input_dim)

        self.protein_fc1 = KANLinear(self.input_dim, self.hid_dim)
        self.protein_fc2 = KANLinear(self.hid_dim, self.hid_dim)
        self.protein_fc4 = KANLinear(self.hid_dim, self.hid_dim)
        self.protein_fc3 = KANLinear(self.hid_dim, self.input_dim)

        self.lin1 = KANLinear(2 * self.input_dim, self.output_dim)

        self.output = KANLinear(self.output_dim, 1)

    def feature_interaction(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.lin1(x)
        return x

    def forward(self, x1, x2):
        x1 = self.input_drug(x1)
        x2 = self.input_protein(x2)

        x1 = self.SE_layer1(x1)
        x2 = self.SE_layer2(x2)

        x1 = self.drug_fc1(x1).relu()
        x1 = self.drug_fc2(x1).relu()
        x1 = self.drug_fc4(x1).relu()
        x1 = self.drug_fc3(x1).relu()

        x2 = self.protein_fc1(x2).relu()
        x2 = self.protein_fc2(x2).relu()
        x2 = self.protein_fc4(x2).relu()
        x2 = self.protein_fc3(x2).relu()

        x = self.feature_interaction(x1, x2)
        out = self.output(x)
        return torch.sigmoid(out)


class Light_SE(torch.nn.Module):
    def __init__(self):
        super(Light_SE, self).__init__()
        self.lin1 = torch.nn.Linear(1, 1)

    def forward(self, inputs):
        x = torch.mean(inputs, dim=-1).unsqueeze(-1)
        x = self.lin1(x)
        x = torch.softmax(x, 1)
        out = inputs * x
        return inputs + out
