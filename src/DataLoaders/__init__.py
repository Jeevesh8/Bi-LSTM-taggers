from src.DataLoaders.xml import load_xml_data

def get_dataloaders(config):

    train_data_loader = load_xml_data(config, split='train/')

    valid_data_loader = load_xml_data(config, split='valid/')

    test_data_loader = load_xml_data(config, split='test/')
    
    return train_data_loader, valid_data_loader, test_data_loader