class Config:
    def __init__(self):
        self.training = True
        self.embedding_name = 'facebook/muppet-roberta-base'
        self.embedding_dropout = 0.3
        self.hidden_num = 300
        self.linear_dropout = 0.1
        self.linear_bias = 1
        self.linear_activation = 'relu'
        self.adapter_learning_rate = 1e-4
        self.num_train_epochs = 5
        self.learning_rate = 2e-5
        self.batch_size=16
        self.max_input_length = 512
        self.adapter_weight_decay = 1e-4
        self.weight_decay = 1e-3
        self.grad_clipping = 1
        
# configuration
config = Config()