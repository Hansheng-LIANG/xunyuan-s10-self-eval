import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from xunyuan_eval.scoring import (add_report_score, coefficient, evaluate, score_route,
                                 score_video, validate_reference)
from xunyuan_eval.io import FormatError, read_json, write_json


def sample(t, x=0, y=0, yaw=0, speed=0):
    # Four corners bound the entire synthetic rectangular footprint.
    points = [[x + dx, y + dy, 0.2] for dx in [-0.2, 0.2] for dy in [-0.1, 0.1]]
    return {"time_s": round(t, 8), "position_m": [x, y, 0.5], "yaw_rad": yaw,
            "roll_rad": 0, "pitch_rad": 0, "speed_mps": speed, "yaw_rate_rps": 0,
            "envelope_points_m": points, "forbidden_contacts": 0, "out_of_bounds": False,
            "action_valid": True, "intervention": False}


def rows_for_route(route):
    rows=[sample(0,*route['path_xy'][0])]
    index=0
    for a,b in zip(route['path_xy'],route['path_xy'][1:]):
        n=max(1,math.ceil(math.dist(a,b)/.015))
        for i in range(1,n+1):
            index+=1;u=i/n
            rows.append(sample(index*.005,a[0]+u*(b[0]-a[0]),y=a[1]+u*(b[1]-a[1]),speed=3))
    return rows


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.ref = validate_reference(read_json(Path(__file__).resolve().parents[1] / 'challenge/reference.json'))
        route = cls.ref['routes'][0]
        cls.records = {'reference_id': cls.ref['reference_id'], 'routes': [], 'disturbance': [], 'friction': []}
        attempts = [('normal', 0, .8, 'routes')]
        attempts += [(f'push_{f}', f, .8, 'disturbance') for f in cls.ref['force_levels_N']]
        attempts += [(f'mu_{mu}', 0, mu, 'friction') for mu in cls.ref['friction_levels']]
        duration = cls.ref['disturbance']['pulse_duration_s']
        for name, force, mu, group in attempts:
            path = cls.root / (name + '.jsonl')
            with path.open('w') as stream:
                for original in rows_for_route(route):
                    row = copy.deepcopy(original)
                    active = any(t + 1e-9 < row['time_s'] <= t + duration + 1e-9 for t in [1, 3, 5])
                    row.update(force_world_N=[0, force if active else 0, 0], ground_contact_friction=[mu])
                    stream.write(json.dumps(row) + '\n')
            item = {'route_id': route['id'], 'trace': path.name, 'termination': 'completed'}
            if force:
                item.update(force_N=force, force_record=name + '.json')
                write_json(cls.root / item['force_record'], {'force_N': force, 'pulses': [
                    {'start_s': t, 'duration_s': duration, 'force_world_N': [0, force, 0]} for t in [1, 3, 5]]})
            if group == 'friction':
                item['friction'] = mu
            cls.records[group].append(item)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_one_perfect_map_scores_six_plus_two_plus_two(self):
        result = evaluate(self.records, self.ref, self.root)
        self.assertEqual(result['navigation_control_score'], 10)
        self.assertEqual(result['scores']['disturbance']['push']['score'], 2)
        self.assertEqual(result['scores']['disturbance']['friction']['score'], 2)

    def test_highest_force_failure_keeps_lower_force_credit(self):
        run = copy.deepcopy(self.records)
        run['disturbance'][0]['termination'] = 'safety_stop'
        result = evaluate(run, self.ref, self.root)
        self.assertEqual(result['scores']['disturbance']['push']['score'], 1.5)
        self.assertEqual([r['force_N'] for r in result['details']['disturbance'][:4]], self.ref['force_levels_N'])
        self.assertEqual([r['score'] for r in result['details']['disturbance'][:4]], [0, .5, .5, .5])

    def test_wrong_actual_friction_cannot_earn_friction_points(self):
        run = copy.deepcopy(self.records)
        # Claim the .1 run as .2 and remove the genuine .2 record.
        run['friction'] = [dict(run['friction'][0], friction=.2)]
        result = evaluate(run, self.ref, self.root)
        self.assertEqual(result['scores']['disturbance']['friction']['score'], 0)
        self.assertFalse(result['details']['friction'][1]['attempt']['friction_record_verified'])

    def test_force_and_friction_orders_are_validated(self):
        for field in ['force_levels_N', 'friction_levels']:
            ref = copy.deepcopy(self.ref); ref[field] = ref[field][::-1]
            with self.assertRaises(FormatError):
                validate_reference(ref)


ROOT = Path(__file__).resolve().parents[1]


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref=validate_reference(json.loads((ROOT/'challenge/reference.json').read_text()))
        cls.route=cls.ref['routes'][0]
        cls.rows=rows_for_route(cls.route)

    def test_five_routes_have_correct_outbound_and_reverse_instructions(self):
        for route in self.ref['routes']:
            task=json.loads((ROOT/'challenge'/route['task']).read_text())
            reverse=[{'left':'right','right':'left','straight':'straight'}[t] for t in route['outbound_turns'][::-1]]
            self.assertEqual([t['turn'] for t in task['return']],reverse)
            self.assertEqual(len(task['outbound']),5)
            self.assertEqual(route['path_xy'][0],route['path_xy'][-1])

    def test_exact_path_and_return_scores_six(self):
        result=score_route(self.rows,self.route,self.ref['rules'],'completed')
        self.assertEqual(result['score'],6)
        self.assertTrue(result['passed'])

    def test_outbound_only_has_no_return_credit(self):
        goal=self.route['path_xy'][len(self.route['path_xy'])//2]
        end=next(i for i,row in enumerate(self.rows) if math.dist(row['position_m'][:2],goal)<1e-8)
        result=score_route(self.rows[:end+1],self.route,self.ref['rules'],'timeout')
        self.assertEqual(result['score'],5)
        self.assertEqual(result['parts'][-1]['score'],0)
        self.assertFalse(result['passed'])

    def test_fifth_turn_scores_before_final_approach(self):
        fifth_exit=self.route['path_xy'][10]
        end=next(i for i,row in enumerate(self.rows) if math.dist(row['position_m'][:2],fifth_exit)<1e-8)
        result=score_route(self.rows[:end+1],self.route,self.ref['rules'],'timeout')
        self.assertEqual(result['score'],5)
        self.assertEqual(result['parts'][-1]['score'],0)

    def test_deviation_deductions_have_distinct_bands(self):
        bands=self.ref['rules']['deviation_bands']
        self.assertEqual([coefficient(d,bands) for d in [.10,.20,.40,.60]],[1,.75,.5,0])
        self.assertEqual([coefficient(d,bands) for d in [.15,.30,.50]],[1,.75,.5])

    def test_collision_invalidates_current_route(self):
        rows=copy.deepcopy(self.rows)
        rows[-2]['forbidden_contacts']=1
        result=score_route(rows,self.route,self.ref['rules'],'completed')
        self.assertEqual(result['score'],0)
        self.assertEqual(result['reason'],'collision')

    def test_spawn_at_finish_and_teleport_do_not_complete_route(self):
        rows=copy.deepcopy(self.rows[:2]);rows[1]['position_m']=self.rows[-1]['position_m']
        result=score_route(rows,self.route,self.ref['rules'],'completed')
        self.assertEqual(result['score'],0)
        rows[1]['position_m']=[40,40,.5]
        result=score_route(rows,self.route,self.ref['rules'],'completed')
        self.assertEqual(result['reason'],'position_jump')

    def test_invalid_weights_and_extra_routes_are_rejected(self):
        for key,value in [('weights',{'route':35,'disturbance':15,'recognition':20,'report':30}),('routes',self.ref['routes']*2)]:
            ref=copy.deepcopy(self.ref);ref[key]=value
            with self.assertRaises(FormatError):validate_reference(ref)


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.ref={'events':[{'id':'test01/E01','video_id':'test01','start_s':1,'end_s':2,'readable':True,'kind':'digits','answer':'00127',
                            'boxes':[{'time_s':1,'box_xyxy':[.4,.4,.6,.6]}]}]}
        self.pred={'detections':[{'video_id':'test01','time_s':1,'box_xyxy':[.4,.4,.6,.6],'status':'ok','value':'00127','unit':None,'confidence':.8}]}

    def test_correct_reading_gets_points_wrong_reading_gets_zero(self):
        self.assertEqual(score_video(self.pred,self.ref)['score'],20)
        self.pred['detections'][0]['value']='127'
        self.assertEqual(score_video(self.pred,self.ref)['score'],0)

    def test_many_guesses_use_highest_confidence_not_any_correct_answer(self):
        wrong=copy.deepcopy(self.pred['detections'][0]);wrong.update(value='00000',confidence=.9)
        self.pred['detections'].append(wrong)
        self.assertEqual(score_video(self.pred,self.ref)['score'],0)

    def test_readings_below_confidence_threshold_are_skipped(self):
        detection = self.pred['detections'][0]
        detection['confidence'] = .49
        self.assertEqual(score_video(self.pred, self.ref)['events'][0]['reason'], 'missing')
        detection['confidence'] = .5
        self.assertEqual(score_video(self.pred, self.ref)['correct'], 1)

    def test_detection_needs_correct_time_and_location(self):
        for update in [{'time_s':3},{'box_xyxy':[0,0,.1,.1]}]:
            pred=copy.deepcopy(self.pred);pred['detections'][0].update(update)
            self.assertEqual(score_video(pred,self.ref)['score'],0)

    def test_unreadable_needs_a_reason(self):
        self.ref['events'][0]['readable']=False
        self.pred['detections'][0].update(status='abstain')
        self.assertEqual(score_video(self.pred,self.ref)['score'],0)
        self.pred['detections'][0]['reason']='blurred'
        self.assertEqual(score_video(self.pred,self.ref)['score'],20)

    def test_training_feedback_can_show_answers_and_test_feedback_omits_them(self):
        self.assertNotIn('expected_value',score_video(self.pred,self.ref)['events'][0])
        feedback=score_video(self.pred,self.ref,include_answers=True)
        self.assertEqual(feedback['events'][0]['expected_value'],'00127')
        self.assertEqual(feedback['events'][0]['prediction']['value'],'00127')

    def test_temperature_tolerance_and_missing_events(self):
        self.ref['events'][0].update(kind='temperature_display',answer='31.4')
        self.pred['detections'][0].update(value='31.5',unit='C')
        self.assertEqual(score_video(self.pred,self.ref)['score'],20)
        self.pred['detections'][0]['value']='31.6'
        self.assertEqual(score_video(self.pred,self.ref)['score'],0)
        self.assertEqual(score_video({'detections':[]},self.ref)['score'],0)


class ManualReportTests(unittest.TestCase):
    def test_manual_points_are_added_to_seventy(self):
        report={'automated_score':70,'scores':{},'pending_manual':{'report':30}}
        result=add_report_score(report,27)
        self.assertEqual(result['final_score'],97)
        self.assertEqual(result['maximum_score'],100)
        self.assertEqual(result['pending_manual'],{})
        self.assertEqual(report['pending_manual'],{'report':30})

    def test_report_points_must_be_finite_and_within_thirty(self):
        report={'automated_score':70}
        for score in [-1,31,True,float('nan'),float('inf')]:
            with self.subTest(score=score):
                with self.assertRaises(FormatError):add_report_score(report,score)


if __name__ == '__main__':
    unittest.main()
