from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from urbansound_segment_task.edge_v2.models.lightgbm_cross_fold import (
    CROSS_FOLD_SCHEMA_VERSION, _fold_result_valid, build_aggregate,
    load_split_manifests, validate_resume_manifest,
)
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


ROOT = Path(__file__).resolve().parents[2]


def per_class(class_id: int, value: float):
    return {"class_id":class_id,"precision":value,"recall":value,"f1":value,"support":10}


def fold_result(fold: int, value: float):
    matrix = [[0]*10 for _ in range(10)]
    for index in range(10): matrix[index][index]=10
    metrics = {
        "accuracy":value,"macro_f1":value,"prediction_count":100,
        "class_order":list(range(10)),"per_class":[per_class(i,value-(i/1000)) for i in range(10)],
        "confusion_matrix":matrix,
    }
    return {
        "test_fold":fold,"validation_fold":fold%10+1,
        "test_metrics":{
            "segment":dict(metrics),
            "clip":{**metrics,"evaluable_clip_count":90,"excluded_zero_segment_clip_count":2},
        },
        "training":{"duration_seconds":float(fold),"after_training_peak_rss_bytes":1000+fold},
        "artifacts":{"model":{"size_bytes":100+fold,"sha256":str(fold)*64}},
    }


RUN_MANIFEST={
    "run_identity_sha256":"a"*64,"cache_identity":"b"*64,"config_sha256":"c"*64,"threads":8,
    "class_order":[{"class_id":i,"class_name":f"class-{i}"} for i in range(10)],
}


class LightgbmCrossFoldTests(unittest.TestCase):
    def test_real_manifests_use_each_test_fold_and_rotating_validation_once(self) -> None:
        manifests=load_split_manifests(ROOT/'results/split_manifests/official-rotating-v1')
        self.assertEqual(sorted(manifests),list(range(1,11)))
        self.assertEqual([manifests[i]['validation_fold'] for i in range(1,11)],[2,3,4,5,6,7,8,9,10,1])

    def test_changed_split_manifest_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=ROOT/'results/split_manifests/official-rotating-v1'
            for path in source.glob('*.json'):
                (root/path.name).write_bytes(path.read_bytes())
            changed=json.loads((root/'test-fold-1.json').read_text(encoding='utf-8'))
            changed['training_folds']=[4,5,6,7,8,9,10]
            (root/'test-fold-1.json').write_text(json.dumps(changed),encoding='utf-8')
            with self.assertRaises(ValueError): load_split_manifests(root)

    def test_aggregate_statistics_mean_std_min_max_median(self) -> None:
        result=build_aggregate([fold_result(i,i/10) for i in range(1,11)],RUN_MANIFEST)
        stats=result['aggregate']['clip_accuracy']
        self.assertAlmostEqual(stats['mean'],0.55)
        self.assertAlmostEqual(stats['median'],0.55)
        self.assertEqual(stats['minimum_fold'],1)
        self.assertEqual(stats['maximum_fold'],10)
        self.assertGreater(stats['population_standard_deviation'],0)

    def test_per_class_and_pooled_secondary_are_labeled(self) -> None:
        result=build_aggregate([fold_result(i,0.8) for i in range(1,11)],RUN_MANIFEST)
        self.assertEqual(len(result['per_class_aggregate']['classes']),10)
        self.assertEqual(len(result['per_class_aggregate']['strongest_three_by_mean_clip_f1']),3)
        self.assertEqual(result['pooled_secondary']['clip']['label'],'pooled_secondary')
        self.assertEqual(result['pooled_secondary']['segment']['prediction_count'],1000)

    def test_zero_segment_denominators_are_aggregated(self) -> None:
        result=build_aggregate([fold_result(i,0.8) for i in range(1,11)],RUN_MANIFEST)
        self.assertEqual(result['zero_segment_clips']['total'],20)

    def test_missing_fold_cannot_produce_success(self) -> None:
        result=build_aggregate([fold_result(i,0.8) for i in range(1,10)],RUN_MANIFEST)
        self.assertEqual(result['status']['outcome'],'incomplete')
        self.assertEqual(result['missing_folds'],[10])

    def test_all_ten_folds_produce_versioned_success(self) -> None:
        result=build_aggregate([fold_result(i,0.8) for i in range(1,11)],RUN_MANIFEST)
        self.assertEqual(result['schema_version'],CROSS_FOLD_SCHEMA_VERSION)
        self.assertEqual(result['status']['outcome'],'success')
        serialized=json.dumps(result,sort_keys=True)
        self.assertEqual(json.loads(serialized)['schema_version'],CROSS_FOLD_SCHEMA_VERSION)
        self.assertNotIn('/Users/',serialized)
        self.assertNotIn('dataset_root',serialized)

    def test_resume_manifest_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError,'resume run identity'):
            validate_resume_manifest({'run_identity_sha256':'a'},{'run_identity_sha256':'b'})

    def test_valid_fold_hash_and_model_are_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=root/'model.txt'; model.write_text('model',encoding='utf-8')
            config=root/'training-config.json'; config.write_text('{}',encoding='utf-8')
            per_class=root/'per-class-metrics.csv'; per_class.write_text('class_id\n',encoding='utf-8')
            def artifact(path: Path):
                return {'relative_path':path.name,'size_bytes':path.stat().st_size,'sha256':streaming_file_sha256(path)}
            result={
                'run_identity_sha256':'a'*64,'status':{'outcome':'success'},
                'artifacts':{'model':artifact(model),'config':artifact(config),'per_class':artifact(per_class)},
            }
            result_path=root/'fold-result.json'
            result_path.write_text(json.dumps(result),encoding='utf-8')
            (root/'fold-result.sha256').write_text(streaming_file_sha256(result_path),encoding='ascii')
            self.assertTrue(_fold_result_valid(root,'a'*64))
            config.write_text('changed',encoding='utf-8')
            self.assertFalse(_fold_result_valid(root,'a'*64))


if __name__=='__main__': unittest.main()
