"""Correct class names for ImageNet-R.

Bug this fixes: `utils/des_imagenetr.json` contains descriptions for all *1000 ImageNet-1k* classes,
in ImageNet-1k order (tench, goldfish, great white shark, tiger shark, ...). The dataset used
`list(json.keys())`, so ImageNet-R label i (i-th wnid folder, sorted) was named after ImageNet-1k
class i: goldfish -> "tench", great white shark -> "goldfish", ..., acorn -> "Scottish Terrier".
MoE-Adapter classifies with CLIP text embeddings of these names, so almost every class had a wrong
name. (The data manager also printed "200 1000" = #classes vs #names.)

ImageNet-R's 200 classes are a subset of ImageNet-1k. Below are the official wnids (sorted, i.e. the
label order of torchvision's ImageFolder) and their ImageNet-1k indices (hendrycks/imagenet-r).
"""

IMAGENET_R_WNIDS = [
    'n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433',
    'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264',
    'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000',
    'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558',
    'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240',
    'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433',
    'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166',
    'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958',
    'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022',
    'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441',
    'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443',
    'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049',
    'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495',
    'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020',
    'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440',
    'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185',
    'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010',
    'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258',
    'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504',
    'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076',
    'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704',
    'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348',
    'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990',
    'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592',
    'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677',
]

IMAGENET_R_IN1K_INDICES = [
    1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107,
    113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199,
    203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277,
    281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337,
    338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414,
    425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558,
    570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776,
    779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934,
    936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988,
]

assert len(IMAGENET_R_WNIDS) == len(IMAGENET_R_IN1K_INDICES) == 200


def imagenet_r_class_names(folder_names, in1k_names):
    """Return one name per ImageNet-R class, aligned with ImageFolder's labels.

    folder_names -- ImageFolder(...).classes (sorted class-folder names, i.e. label order)
    in1k_names   -- the 1000 ImageNet-1k class names in class-index order
                    (keys of utils/des_imagenetr.json)
    """
    in1k_names = list(in1k_names)
    if len(in1k_names) != 1000 or in1k_names[1] != 'goldfish' or in1k_names[988] != 'acorn':
        raise ValueError('Expected the 1000 ImageNet-1k class names in index order '
                         '(got {} names, starting {}).'.format(len(in1k_names), in1k_names[:3]))
    wnid_to_name = {w: in1k_names[i] for w, i in zip(IMAGENET_R_WNIDS, IMAGENET_R_IN1K_INDICES)}
    folder_names = list(folder_names)
    if len(folder_names) != 200:
        raise ValueError('ImageNet-R should have 200 class folders, found {}.'.format(len(folder_names)))
    if all(f in wnid_to_name for f in folder_names):
        return [wnid_to_name[f] for f in folder_names]
    # Some copies of ImageNet-R use readable folder names instead of wnids.
    if not any(f.startswith('n') and f[1:].isdigit() for f in folder_names):
        return [f.replace('_', ' ') for f in folder_names]
    unknown = [f for f in folder_names if f not in wnid_to_name][:5]
    raise ValueError('Unrecognised ImageNet-R class folders, e.g. {}.'.format(unknown))
