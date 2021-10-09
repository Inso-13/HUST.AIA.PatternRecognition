class OVO:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.classes_list = self.generate_list()
        self.PLA_models = [None for i in len(self.classes_list)]

    def generate_list(self):
        n = self.num_classes
        temp_list = []
        for i in range(n):
            for j in range(i+1, n):
                temp_list.append([i, j])
        return temp_list

    def 
