import os.path as osp
import time
import sys 
sys.path.append("/home/zhangyuekun/GeoTransformer")
sys.path.append("/home/zhangyuekun/GeoTransformer/experiments")


from geotransformer.engine import SingleTester
from geotransformer.utils.common import get_log_string

# from dataset import test_data_loader
from network.weight_net.dataset import test_data_loader
from network.weight_net.config import make_cfg
from network.weight_net.model import create_model
from network.weight_net.loss import Evaluator
from network.weight_net.vis import draw_node_correspondences2,save_final_cor
import numpy as np

class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        # self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        # self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        # print(output_dict["ref_points_c"].shape)
        # print("output_dict info:")
        # for i in output_dict.items():
        #     # print(i.cpu().numpy().shape)
        #     print(i[0])
        #     print(i[1].shape)
            # print(i[1])
        # data_list = [x for x in data_list if x['label'] in self.class_indices]
        ref_nodes = output_dict["ref_corr_points"]
        src_nodes = output_dict["src_corr_points"]
        print("let's save final cor")
        np.save('ref_points_f.npy',np.asarray(output_dict["ref_points_f"].cpu()))
        save_final_cor(output_dict["ref_corr_points"].cpu(),output_dict["src_corr_points"].cpu(),output_dict["corr_scores"].cpu(),output_dict["estimated_transform"].cpu())
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    # def summary_string(self, iteration, data_dict, output_dict, result_dict):
    #     message = get_log_string(result_dict=result_dict)
    #     message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
    #     return message


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    print("run: ")
    tester.run()


if __name__ == '__main__':
    main()
# --snapshot=/home/zhangyuekun/GPV_Pose/network/weight_net/weights/geotransformer-modelnet.pth.tar