from src.DataLoaders.xml import load_xml_data

def get_dataloaders(config):

    train_data_loader = load_xml_data(config, split='train/')

    valid_data_loader = load_xml_data(config, split='valid/')

    test_data_loader = load_xml_data(config, split='test/')
    
    config['total_steps'] = len([0 for thread in train_data_loader.thread_generator()])

    print("Total training steps: ", config['total_steps'])    
    
    return train_data_loader, valid_data_loader, test_data_loader, config