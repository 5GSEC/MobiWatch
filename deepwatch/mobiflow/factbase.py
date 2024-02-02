from .mobiflow import *
import time

class FactBase:
    def __init__(self):
        self.facts = {}
        self.bs_id_counter = 0
        self.bs_name_map = {}

    def add_bs(self, bs: BS):
        if not self.facts.keys().__contains__(bs.bs_id):
            bs.ts = time.time() * 1000
            bs.bs_id = self.bs_id_counter
            self.bs_name_map[bs.name] = bs.bs_id
            self.bs_id_counter += 1
            self.facts[bs.bs_id] = bs
        else:
            self.facts[bs.bs_id].ts = time.time()

    def add_ue(self, bsId, ue: UE):
        if not self.facts.keys().__contains__(bsId):
            print("[Error] BS id not exist when trying to add UE: %s" % bsId)
            return
        else:
            ue.ts = time.time() * 1000
            ue.bs_id = bsId
            if self.facts[bsId] is not None:
                self.facts[bsId].add_ue(ue)

    def get_bs_index_by_name(self, bs_name: str):
        if self.bs_name_map.keys().__contains__(bs_name):
            return self.bs_name_map[bs_name]
        else:
            return -1

    # BS related operations
    def get_all_bs(self):
        bss = []
        for bsId in self.facts.keys():
            bss.append(self.facts[bsId])
        return bss

    def get_bs(self, bsId):
        return self.facts.get(bsId)

    def remove_bs(self, bsId):
        if bsId in self.facts.keys():
            del self.facts[bsId]
            return True
        else:
            return False

    # UE related operations
    def get_all_ue(self):
        ues = []
        for bsId in self.facts.keys():
            ues += self.facts[bsId].ue
        return ues

    def get_ue(self, rnti: int):
        ues = self.get_all_ue()
        for ue in ues:
            if ue.rnti == rnti:
                return ue
        return None

    def remove_ue(self, rnti: int):
        ue_to_remove = None
        for bsId in self.facts.keys():
            for ue in self.facts[bsId].ue:
                if ue.rnti == rnti:
                    ue_to_remove = ue
                    break
            if ue_to_remove is not None:
                self.facts[bsId].ue.remove(ue_to_remove)
                return True
        return False

