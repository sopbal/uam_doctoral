#!/usr/bin/env python3
# coding: utf-8

#a modified version of this script has been published in the zwmuam /lysin_vocabulary repository as part of the publication:
#Bałdysz S, Nawrot R, Barylski J.2024.“Tear down that wall”—a critical evaluation of bioinformatic resources available for lysin researchers. 
#Appl Environ Microbiol90:e02361-23.https://doi.org/10.1128/aem.02361-23


from copy import copy
from pathlib import Path
from typing import List, Union

PathLike = Union[Path, str]

# dictionary with column numbers for different HMMER3 domtblout file formats
PARSER_DICT = {'hmmsearch': {'prot_id': 0,
                             'hmm_id': 3,
                             'prot_start': 17,
                             'prot_end': 18,
                             'evalue': 6,
                             'score': 13},
               'hmmscan': {'prot_id': 3,
                           'hmm_id': 0,
                           'prot_start': 17,
                           'prot_end': 18,
                           'evalue': 6,
                           'score': 13}}

class Domain:
    """
    An abstract object
    symbolising a single domain
    contains all relevant attributes
    and relevant functions
    """
    def __init__(self, line: str, program: str = 'hmmscan'):
        """
        Object initialisation (creation of the class instance)
        the function assumes that it has a line from hmmscan
        and has to be told otherwise if hmmsearch was used
        """
        split_line = line.split()
        self.prot_id = split_line[PARSER_DICT[program]['prot_id']]
        self.hmm_id = split_line[PARSER_DICT[program]['hmm_id']]
        self.prot_start, self.prot_end = int(split_line[PARSER_DICT[program]['prot_start']]), int(split_line[PARSER_DICT[program]['prot_end']])
        self.evalue = float(split_line[PARSER_DICT[program]['evalue']])
        self.score = float(split_line[PARSER_DICT[program]['score']])
        self.line = line

    def __repr__(self):
        """
        How an instance should look like in the printout etc.
        """
        return f'{self.hmm_id} [{self.prot_start} - {self.prot_end}] ({self.score})'

    def gff(self, id_string: str) -> str:
        """
        Represent a domain as a GFF line
        :param id_string: a short, unique identifier string
        :return: gff-formatted line (no newline (\n) at the end)
        """
        attributes = {'ID': id_string,
                      'e_value': self.evalue,
                      'HMM': self.hmm_id}
        attribute_string = ';'.join(f'{k}={v}' for k, v in attributes.items())
        return '\t'.join([self.prot_id,
                          'HMMER3',
                          'domain',
                          str(self.prot_start),
                          str(self.prot_end),
                          str(self.score),
                          '.', '.',
                          attribute_string])

    def seq_length(self) -> int:
        """
        Calculate length of the domain in AA
        :return: length of the domain in AA
        """
        return self.prot_end - self.prot_start + 1

    def overlaps(self, domain: 'Domain') -> bool:
        """
        Check if any of the two analysed domains overlap
        on more than 50% of its length with the other
        :param domain:
        :return: do these domains overlap?
        """
        if self.prot_end >= domain.prot_start and self.prot_start <= domain.prot_end:
            overlap_start = max(self.prot_start, domain.prot_start)
            overlap = domain.prot_end - overlap_tart + 1
            if overlap < 1:
                raise NotImplementedError()
            if any([overlap > e.seq_length() * 0.5 for e in (self, domain)]):
                return True
        elif domain.prot_end >= self.prot_start and domain.prot_start <= self.prot_end:
            return domain.overlaps(self)
        return False


def resolve_overlap(domain_list: List[Domain]) -> Domain:
    """
    Given a list of domains choose the top-scoring one
    :param domain_list: list of two or more domains
    :return: Top scoring domain from the cluster
    """
    ranking = sorted(domain_list, key=lambda dom: dom.score, reverse=True)
    return ranking[0]


def clean_file(file_path: PathLike, max_eval: float = 1e-5):
    """
    Choose only the best (locally) domains for all proteins in a single file
    save a '.culling.gff', '.culling.txt' and '.culling.domtblout'
    files in the parent folder of the input file
    :param file_path: system path to a file
          (please avoid any non standard characters
           including whitespaces)
    """
    file_path = Path(file_path)
    with file_path.open() as inpt:
        protein_dict = {}
        discarded, kept = 0, 0
        for line in inpt:
            if line.strip() and not line.startswith('#'):
                domain = Domain(line, program='hmmscan')
                if domain.evalue < max_eval:
                    if domain.prot_id not in protein_dict:
                        protein_dict[domain.prot_id] = []
                    protein_dict[domain.prot_id].append(domain)
                    kept += 1
                else:
                    discarded += 1
            else:
                print(f'Line skipped: {line}')
    print(f'E-value filter: {max_eval} ({kept} hits kept, {discarded} discarded)')

    gff_path = Path(file_path.as_posix() + '.culling.gff')
    summary_path = Path(file_path.as_posix() + '.culling.txt')
    domtblout_path = Path(file_path.as_posix() + '.culling.domtblout')

    with gff_path.open('w') as gff:
        with summary_path.open('w') as summary:
            with domtblout_path.open('w') as domtblout:
                domain_index = 1
                for protein, domains in protein_dict.items():
                    #print(protein)
                    domains.sort(key=lambda dom: dom.prot_start)
                    starting_domains = copy(domains)
                    filtered_domains = [domains.pop(0)]
                    while domains:
                        new_domain = domains.pop(0)
                        overlap_found = False
                        for i, old_domain in enumerate(filtered_domains):
                            if new_domain.overlaps(old_domain):
                                filtered_domains[i] = resolve_overlap([new_domain, old_domain])
                                overlap_found = True
                                break
                        if not overlap_found:
                            filtered_domains.append(new_domain)
                    if len(filtered_domains) != len(starting_domains):
                        print(f'{starting_domains} -> {filtered_domains}')
                    for domain in filtered_domains:
                        gff.write(domain.gff(str(domain_index)) + '\n')
                        domtblout.write(domain.line)
                        domain_index += 1
                    hmm_string = '; '.join([d.hmm_id for d in filtered_domains])
                    summary.write(f'{protein}\t{hmm_string}\n')


if __name__ == '__main__':
    # actually run anything
    input_path = Path('.../example_hmmscan_results')
    clean_file(input_path)
