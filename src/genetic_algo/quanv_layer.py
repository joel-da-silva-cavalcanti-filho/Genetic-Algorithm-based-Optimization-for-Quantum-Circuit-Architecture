import torch.nn
import cmath
import math
from ansatz_simulation_class import AnsatzSimulation
from sklearn.feature_extraction import image

class QuanvLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, patch_size, mode="both", chromossome):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.chromossome = chromossome
        self.mode = mode
        self.qlayer = AnsatzSimulation(n_qubits)
        #self.dev = qml.device("lightning.gpu", wires=n_qubits)

        # Como definir o meu circuito em termos de tensores?
        """"

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method="adjoint")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        
        """
    
        """


        """

    def quanv_2d(self, x):
        patches = image.extract_patches_2d(x, (self.patch_size, self.patch_size))
        outputs = []
    
        for patch in patches:
            expval = self.qlayer.simulate_circuit(torch(patch), self.chromossome)
            outputs.append(expval)

        return outputs

    def quanv1d_horizontal(self, x):
        B, C, H, W = x.shape
        out_w = W // self.patch_size
        outputs = []

        for i in range(H): 
            row = x[:, 0, i, :]              # [B, W]
            patches = []
            for j in range(0, W, self.patch_size):
                patch = row[:, j:j+self.patch_size]  # [B, patch_size]
                if patch.shape[1] != self.n_qubits:
                    continue
                q_out = self.qlayer(patch)           # [B, n_qubits]
                patches.append(q_out)
            patches = torch.stack(patches, dim=2)    # [B, n_qubits, out_w]
            outputs.append(patches)

        outputs = torch.stack(outputs, dim=2)        # [B, n_qubits, H, out_w]
        return outputs

    def quanv1d_vertical(self, x):
        B, C, H, W = x.shape
        out_h = H // self.patch_size
        outputs = []

        for j in range(W): 
            col = x[:, 0, :, j]              # [B, H]
            patches = []
            for i in range(0, H, self.patch_size):
                patch = col[:, i:i+self.patch_size]  # [B, patch_size]
                if patch.shape[1] != self.n_qubits:
                    continue
                q_out = self.qlayer(patch)           # [B, n_qubits]
                patches.append(q_out)
            patches = torch.stack(patches, dim=2)    # [B, n_qubits, out_h]
            outputs.append(patches)

        outputs = torch.stack(outputs, dim=3)        # [B, n_qubits, out_h, W]
        return outputs
    
    def quanv1d_interlaced(self, h_out, v_out):
        interlaced_tensor = torch.stack((h_out, v_out), dim = 3)
        return interlaced_tensor
    
    def aggregate_sobel(self, h_out, v_out, method):
        if method == "euclidean":
            v_out_transposed = v_out.transpose(2, 3)
            return torch.sqrt(h_out**2 + v_out_transposed**2)
        elif method == "sum":
            return h_out + v_out
        elif method == "abs_sum":
            return torch.abs(h_out) + torch.abs(v_out)
        else:
            raise ValueError(f"Invalid aggregation method: {method}")
    

    def forward(self, x):
        if self.mode == "horizontal":
            return self.quanv1d_horizontal(x)
        elif self.mode == "vertical":
            return self.quanv1d_vertical(x)
        elif self.mode == "both":
            print("Beginning horizontal processing...")
            h_out = self.quanv1d_horizontal(x)
            print("Beginning vertical processing...")
            v_out = self.quanv1d_vertical(x)
            print("Aggregation...")
            return self.aggregate_sobel(h_out, v_out, method="euclidean") 
        elif self.mode == "interlaced":
            print("Beginning horizontal processing...")
            h_out = self.quanv1d_horizontal(x)
            print("Beginning vertical processing...")
            v_out = self.quanv1d_vertical(x)
            print("Interlacing...")
            return self.quanv1d_interlaced(h_out, v_out)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
