import pytest
import torch
import torch.nn as nn
import torch.optim as optim

@pytest.fixture(scope="session")
def sequential_net():
    """元のコードと同じ構造 (入力4, 出力4) のモデルを生成"""
    net = torch.nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 4) 
    )
    return net.to(torch.device("cpu"))

@pytest.fixture(scope="session")
def dummy_data():
    """ダミー入力データ (バッチサイズ 5, 特徴量 4) とラベル (0-3)"""
    features = torch.randn(5, 4, dtype=torch.float)
    labels = torch.randint(0, 4, (5,), dtype=torch.long) 
    return features, labels

def test_model_output_shape(sequential_net: nn.Module, dummy_input: torch.Tensor = dummy_data):
    """元のモデルが出力形状 (Batch, 4) を返すことを検証"""
    net = sequential_net.eval()
    with torch.no_grad():
        output: torch.Tensor = net(dummy_input)
    
 
    assert output.shape == torch.Size([7, 8])
    assert output.dtype == torch.float

def test_parameters_update_works(sequential_net, dummy_data):
    """訓練ステップ後にモデルの重みが実際に更新されていることを検証 """
    net = sequential_net.train()
    features, labels = dummy_data
    
 
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
   
    initial_weight = net.weight.clone().detach() 
    
    optimizer.zero_grad()
    loss = criterion(net(features), labels)
    loss.backward()
    optimizer.step()

    final_weight = net.weight.detach()

    assert not torch.equal(initial_weight, final_weight), \
        "勾配計算と最適化ステップが正常に動作し、パラメータが更新されました。" [12]