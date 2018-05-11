import os
import sys

import logging
import argparse

"""
    主程序，使用不同方法计算文本相似度，并对测试集进行检索
"""

class Methods:
    """
    模型方法类
    """

    _methods = []
    _modules = {}
    _funcs = {}

    def parseFunc(self, func):
        func_info = func.split(".")
        if(len(func_info) != 2):
            return None, None
        try:
            _modules = __import__(func_info[0])
            if(not hasattr(_modules, func_info[1])):
                return None, None
            return func_info[0], func_info[1]
        except Exception:
            return None, None

    def register(self, method, func):
        if method not in self._methods:
            module, func_name = self.parseFunc(func)
            if module is None:
                logging.error("Not have the func: %s" % func)
                return False
            self._methods.append(method)
            self._modules[method] = module
            self._funcs[method] = func_name
        else:
            logging.info("The method exist!")

    def unregister(self, method, func):
        if method not in self._methods:
            return
        _methods.remove(method)

        del self._modules[method]
        del self._funcs[method]

    def has(self, method):
        return method in self._methods

    def exe(self, method, **args):
        if not self.has(method):
            logging.error("The method %s not exist" % method)
            return
        module = __import__(self._modules[method])
        func = getattr(module, self._funcs[method])
        func(args)


def init(methods):
    methods.register("doc2vec", "methods.doc2vec")
    methods.register("tfidf_word2vec", "methods.tfidf_word2vec")
    methods.register("lcs", "methods.lcs")
    methods.register("jacarrd", "methods.jaccardSim")
    methods.register("tfidf", "methods.tfidf")


if __name__ == '__main__':

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")
    logging.root.setLevel(logging.INFO)
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, required=True, help="使用的模型")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")

    args, _ = parser.parse_known_args()

    method = args.method

    methods = Methods()
    init(methods)

    if not methods.has(method):
        logger.error("%s方法不能被识别" % method)
        sys.exit(1)

    methods.exe(method, train_data=args.train_data, infer_data=args.test_data, model_path=args.model_path)
