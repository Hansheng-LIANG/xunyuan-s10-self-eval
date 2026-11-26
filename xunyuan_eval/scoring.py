"""Score five routes, four disturbance levels, and video readings."""

import copy
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .io import FormatError, contained, number, read_json, text, vector
from .validation import load_trace


def distance_to_segment(point, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length2 = dx*dx+dy*dy
    u = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length2)) if length2 else 0.
    return math.hypot(point[0]-a[0]-u*dx, point[1]-a[1]-u*dy)


def coefficient(error, bands):
    for band in bands:
        if error <= band['distance_m'] + 1e-9:
            return band['credit']
    return 0.


def score_route(rows, route, rules, termination):
    """Follow ordered checkpoints, measuring the greatest path error per section."""
    points = route['path_xy']
    section_for_edge = route['section_for_edge']
    edge = 0
    errors = [0.]*6
    reached = [False]*6
    stop = None
    last = None
    valid_until = 0.
    for row in rows:
        t = row['time_s']
        if t > rules['timeout_s']:
            stop = 'timeout'; break
        if last is not None:
            dt = t-last['time_s']
            if dt > rules['max_gap_s']+1e-9:
                stop = 'telemetry_gap'; break
            displacement = math.dist(row['position_m'][:2], last['position_m'][:2])
            if displacement > rules['maximum_speed_mps']*dt+rules['step_position_slack_m']:
                stop = 'position_jump'; break
        stop = ('collision' if row['forbidden_contacts'] else
                'out_of_bounds' if row['out_of_bounds'] else
                'illegal_action' if not row['action_valid'] else
                'manual_intervention' if row['intervention'] else
                'fall' if max(abs(row['roll_rad']),abs(row['pitch_rad'])) > rules['fall_angle_rad'] else None)
        if stop:
            break
        point = row['position_m'][:2]
        if edge == 0 and last is None and math.dist(point, points[0]) > rules['start_radius_m']:
            stop = 'wrong_start'; break
        if edge < len(points)-1:
            section = section_for_edge[edge]
            error = distance_to_segment(point, points[edge], points[edge+1])
            if edge and math.dist(point,points[edge]) <= rules['checkpoint_radius_m']+.02:
                error=min(error,distance_to_segment(point,points[edge-1],points[edge]))
            errors[section] = max(errors[section], error)
            if math.dist(point, points[edge+1]) <= rules['checkpoint_radius_m']:
                edge += 1
                if edge == len(points)-1 or section_for_edge[edge] != section:
                    reached[section] = True
        valid_until = t
        last = row
    completed = edge == len(points)-1
    safety_failed = stop in {'collision','out_of_bounds','illegal_action','manual_intervention','fall','position_jump','wrong_start','telemetry_gap'}
    if termination not in {'completed','timeout','crashed','safety_stop'}:
        raise FormatError('Unknown termination')
    if termination in {'crashed','safety_stop'}:
        safety_failed = True
        stop = stop or termination
    parts = []
    for i in range(6):
        credit = coefficient(errors[i],rules['deviation_bands']) if reached[i] else 0.
        if safety_failed:
            credit = 0.
        parts.append({'id':f'J{i+1:02d}' if i<5 else 'return','reached':reached[i],
                      'maximum_deviation_m':errors[i],'score':credit,'maximum':1})
    return {'route_id':route['id'],'score':sum(p['score'] for p in parts),'maximum':6,
            'completed':completed and not safety_failed and stop is None and termination=='completed','passed':completed and not safety_failed and stop is None and termination=='completed' and max(errors)<=rules['disturbance_path_limit_m'],
            'reason':stop or ('completed' if completed and termination=='completed' else termination if termination!='completed' else 'route_incomplete'),
            'last_checkpoint':edge,'scored_until_s':valid_until,'parts':parts}


def intersection_over_union(a, b):
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def validate_video_reference(reference):
    if not isinstance(reference, dict) or not isinstance(reference.get('events'), list) or not reference['events']:
        raise FormatError('Video reference needs events')
    try:
        tolerance = Decimal(str(reference.get('temperature_tolerance_C', '.1')))
    except InvalidOperation as exc:
        raise FormatError('Invalid temperature tolerance') from exc
    if not tolerance.is_finite() or tolerance < 0:
        raise FormatError('Invalid temperature tolerance')
    iou = number(reference.get('minimum_iou', .2), 'minimum_iou', 0)
    if not 0 < iou <= 1:
        raise FormatError('minimum_iou must be between zero and one')
    if number(reference.get('box_time_tolerance_s', .15), 'box time tolerance', 0) == 0:
        raise FormatError('Box time tolerance must be positive')
    seen = set()
    for event in reference['events']:
        if not isinstance(event, dict):
            raise FormatError('Invalid video event')
        identity = text(event.get('id'), 'event id')
        if identity in seen:
            raise FormatError('Duplicate video event ID')
        seen.add(identity)
        text(event.get('video_id'), 'video id')
        start = number(event.get('start_s'), 'event start', 0)
        if number(event.get('end_s'), 'event end', 0) <= start:
            raise FormatError('Invalid event interval')
        if type(event.get('readable')) is not bool or event.get('kind') not in {'digits', 'temperature_display'}:
            raise FormatError('Invalid video event label')
        if event['readable']:
            answer = text(event.get('answer'), 'event answer')
            if event['kind'] == 'temperature_display':
                try:
                    if not Decimal(answer).is_finite():
                        raise InvalidOperation
                except InvalidOperation as exc:
                    raise FormatError('Invalid temperature answer') from exc
        frames = event.get('boxes')
        if not isinstance(frames, list) or not frames:
            raise FormatError('Video event needs observed boxes')
        previous = -1.
        for frame in frames:
            t = number(frame.get('time_s'), 'box time', 0)
            if t <= previous or not start <= t < event['end_s']:
                raise FormatError('Invalid box timestamps')
            previous = t
            box = frame.get('box_xyxy')
            if not isinstance(box, list) or len(box) != 4:
                raise FormatError('Invalid reference box')
            for value in box:
                number(value, 'box coordinate')
            if not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
                raise FormatError('Invalid reference box bounds')
    return reference


def event_candidates(event, detections, tolerance, minimum_iou):
    for row in detections:
        t = row['time_s']
        if not event['start_s'] <= t < event['end_s']:
            continue
        frame = min(event['boxes'], key=lambda frame: abs(frame['time_s'] - t))
        if abs(frame['time_s'] - t) <= tolerance and intersection_over_union(row['box_xyxy'], frame['box_xyxy']) >= minimum_iou:
            yield row


def reading_matches(event, prediction, tolerance):
    if prediction is None:
        return False
    if not event['readable']:
        reason = prediction.get('reason')
        return prediction['status'] == 'abstain' and isinstance(reason, str) and bool(reason.strip())
    value = prediction.get('value')
    if prediction['status'] != 'ok' or not isinstance(value, str):
        return False
    if event['kind'] == 'digits':
        return value == event['answer'] and prediction.get('unit') is None
    if prediction.get('unit') != 'C':
        return False
    try:
        temperature = Decimal(value)
        return temperature.is_finite() and abs(temperature - Decimal(event['answer'])) <= tolerance
    except InvalidOperation:
        return False


def score_video(predictions, reference, include_answers=False):
    """Match by time and box, then score the most confident reading of each sign."""
    if not isinstance(predictions, dict) or not isinstance(predictions.get('detections'), list):
        raise FormatError('Video predictions need a detections list')
    validate_video_reference(reference)
    by_video = {event['video_id']: [] for event in reference['events']}
    for row in predictions['detections']:
        if not isinstance(row, dict):
            raise FormatError('Detection must be an object')
        if row.get('video_id') not in by_video:
            raise FormatError('Unknown video ID')
        number(row.get('time_s'), 'time_s', 0)
        confidence = number(row.get('confidence'), 'confidence', 0)
        if confidence > 1:
            raise FormatError('Confidence must be at most one')
        box = vector(row.get('box_xyxy'), 4, 'box_xyxy')
        if not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
            raise FormatError('Detection box must be normalized xyxy')
        if row.get('status') not in {'ok', 'abstain'}:
            raise FormatError('Invalid recognition status')
        if confidence >= .5:
            by_video[row['video_id']].append(row)

    events = []
    box_tolerance = reference.get('box_time_tolerance_s', .15)
    minimum_iou = reference.get('minimum_iou', .2)
    temperature_tolerance = Decimal(str(reference.get('temperature_tolerance_C', '.1')))
    for event in reference['events']:
        candidates = event_candidates(event, by_video[event['video_id']], box_tolerance, minimum_iou)
        chosen = max(candidates, key=lambda row: row['confidence'], default=None)
        passed = reading_matches(event, chosen, temperature_tolerance)
        reason = 'missing' if chosen is None else 'correct' if passed else 'incorrect'
        outcome = {'id': event['id'], 'video_id': event['video_id'], 'passed': passed,
                   'reason': reason, 'prediction': chosen}
        if include_answers:
            outcome['expected_status'] = 'ok' if event['readable'] else 'abstain'
            outcome['expected_value'] = event.get('answer') if event['readable'] else None
        events.append(outcome)
    correct = sum(row['passed'] for row in events)
    return {'score': 20 * correct / len(events), 'maximum': 20, 'correct': correct,
            'total': len(events), 'events': events}


def validate_reference(reference):
    if not isinstance(reference, dict):
        raise FormatError('Expected challenge rules')
    text(reference.get('reference_id'), 'reference_id')
    routes = reference.get('routes')
    if not isinstance(routes, list) or len(routes) != 5 or any(not isinstance(r, dict) for r in routes):
        raise FormatError('Expected five routes')
    ids = [text(r.get('id'), 'route id') for r in routes]
    if len(set(ids)) != 5:
        raise FormatError('Expected five unique route IDs')
    if reference.get('weights') != {'route': 30, 'disturbance': 20, 'recognition': 20, 'report': 30}:
        raise FormatError('Expected 30 route + 20 disturbance + 20 recognition + 30 report')
    tiers = reference.get('force_levels_N')
    if not isinstance(tiers, list) or len(tiers) != 4:
        raise FormatError('Expected four decreasing force levels')
    for value in tiers:
        if number(value, 'force_N', 0) == 0:
            raise FormatError('Force must be positive')
    if sorted(set(tiers), reverse=True) != tiers:
        raise FormatError('Expected four decreasing force levels')
    rules = reference.get('rules')
    if not isinstance(rules, dict):
        raise FormatError('Missing route rules')
    for key in ['timeout_s', 'sample_dt_s', 'max_gap_s', 'start_radius_m', 'checkpoint_radius_m',
                'maximum_speed_mps', 'step_position_slack_m', 'fall_angle_rad', 'disturbance_path_limit_m']:
        if number(rules.get(key), key, 0) == 0:
            raise FormatError(key + ' must be positive')
    if rules['max_gap_s'] < rules['sample_dt_s']:
        raise FormatError('max_gap_s must cover a physics step')
    bands = rules.get('deviation_bands')
    if not isinstance(bands, list) or not bands:
        raise FormatError('Missing deviation bands')
    distances = []
    credits = []
    for band in bands:
        if not isinstance(band,dict):raise FormatError('Invalid deviation band')
        distances.append(number(band.get('distance_m'), 'deviation distance', 0))
        credit = number(band.get('credit'), 'deviation credit', 0)
        if credit > 1:
            raise FormatError('Deviation credit cannot exceed one')
        credits.append(credit)
    if distances != sorted(set(distances)) or credits != sorted(credits, reverse=True):
        raise FormatError('Deviation bands must increase in distance and decrease in credit')
    disturbance = reference.get('disturbance', {})
    if not isinstance(disturbance,dict):raise FormatError('Invalid disturbance configuration')
    if disturbance.get('pulses_per_attempt') != 3:
        raise FormatError('Expected three disturbance pulses')
    if number(disturbance.get('pulse_duration_s'), 'pulse duration', 0) == 0:
        raise FormatError('Pulse duration must be positive')
    fractions = disturbance.get('progress_fractions')
    if not isinstance(fractions, list) or len(fractions) != 3:
        raise FormatError('Expected three pulse positions')
    for fraction in fractions:
        if not 0 < number(fraction, 'pulse progress') < 1:
            raise FormatError('Pulse positions must lie inside the route')
    if fractions != sorted(set(fractions)):
        raise FormatError('Pulse positions must be ordered')
    jitter=number(disturbance.get('progress_jitter_fraction',.04),'pulse jitter',0)
    if fractions[0]-jitter<=0 or fractions[-1]+jitter>=1 or any(b-a<=2*jitter for a,b in zip(fractions,fractions[1:])):
        raise FormatError('Pulse jitter must keep positions ordered inside the route')
    for route in routes:
        points = route.get('path_xy')
        sections = route.get('section_for_edge')
        if not isinstance(points, list) or len(points) < 7 or not isinstance(sections, list):
            raise FormatError('Missing route geometry')
        for point in points:
            vector(point, 2, 'route point')
        if any(type(section) is not int for section in sections):
            raise FormatError('Invalid route section')
        if len(sections) != len(points)-1 or sorted(set(sections)) != list(range(6)) or sections != sorted(sections):
            raise FormatError('Route needs five outbound sections and one return section')
        if math.dist(points[0], points[-1]) > 1e-8:
            raise FormatError('Reference route must return to its start')
        if any(math.dist(a, b) < 1e-8 for a, b in zip(points, points[1:])):
            raise FormatError('Route contains a zero-length edge')
    friction = reference.get('friction_levels')
    if not isinstance(friction, list) or len(friction) != 4:
        raise FormatError('需要四档地面摩擦系数')
    for mu in friction:
        if number(mu, 'friction', 0) == 0:
            raise FormatError('摩擦系数必须大于 0')
    if friction != sorted(set(friction)):
        raise FormatError('摩擦测试从最滑到最不滑排列')
    if reference.get('disturbance_weights') != {'push': 10, 'friction': 10}:
        raise FormatError('扰动分为推力 10 分、摩擦 10 分')
    nominal = number(reference.get('nominal_friction'), 'nominal friction', 0)
    if nominal <= 0:
        raise FormatError('正常地面摩擦系数必须大于 0')
    return reference


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_force_record(rows, record, force, reference):
    disturbance = reference['disturbance']
    duration = disturbance['pulse_duration_s']
    if not isinstance(record, dict) or record.get('force_N') != force:
        return False
    pulses = record.get('pulses')
    if not isinstance(pulses, list) or len(pulses) != disturbance['pulses_per_attempt']:
        return False
    previous_end = 0
    for pulse in pulses:
        if not isinstance(pulse, dict):
            return False
        start = pulse.get('start_s')
        vector = pulse.get('force_world_N')
        if not finite_number(start) or start < 0 or start < previous_end - 1e-8:
            return False
        if not isinstance(vector, list) or len(vector) != 3 or not all(map(finite_number, vector)):
            return False
        if abs(math.sqrt(sum(v*v for v in vector)) - force) >= 1e-5 or abs(vector[2]) >= 1e-8:
            return False
        if not finite_number(pulse.get('duration_s')) or abs(pulse['duration_s'] - duration) >= 1e-8:
            return False
        previous_end = start + duration

    impulse = 0
    for previous, row in zip(rows, rows[1:]):
        vector = row.get('force_world_N')
        if not isinstance(vector, list) or len(vector) != 3 or not all(map(finite_number, vector)):
            return False
        t = previous['time_s']
        expected_vector = next((p['force_world_N'] for p in pulses
                                if p['start_s'] - 1e-9 <= t < p['start_s'] + duration - 1e-9), [0, 0, 0])
        if math.dist(vector, expected_vector) > 1e-5:
            return False
        impulse += math.sqrt(sum(v*v for v in vector)) * (row['time_s'] - t)

    expected_impulse = force * len(pulses) * duration
    return abs(impulse - expected_impulse) < force * reference['rules']['sample_dt_s'] * .1


def valid_friction_record(rows, coefficient):
    observed_contact = False
    for row in rows:
        contacts = row.get('ground_contact_friction')
        if not isinstance(contacts, list):
            return False
        if any(not finite_number(mu) or abs(mu - coefficient) >= 1e-6 for mu in contacts):
            return False
        observed_contact |= bool(contacts)
        force = row.get('force_world_N')
        if not isinstance(force, list) or len(force) != 3:
            return False
        if any(not finite_number(v) or abs(v) > 1e-8 for v in force):
            return False
    return observed_contact


def index_attempts(rows, route_ids, field=None, levels=()):
    if not isinstance(rows, list):
        raise FormatError('Attempts must be a list')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('route_id') not in route_ids:
            raise FormatError('Unknown route in attempt')
        key = row['route_id']
        if field:
            level = row.get(field)
            if not finite_number(level) or level not in levels:
                raise FormatError(f'Unknown {field} level')
            key = (key, level)
        if key in indexed:
            raise FormatError('Duplicate route attempt')
        indexed[key] = row
    return indexed


def evaluate(run, reference, root, video_reference=None):
    validate_reference(reference)
    if not isinstance(run, dict) or run.get('reference_id') != reference['reference_id']:
        raise FormatError('Run and reference must identify the same challenge')
    root = Path(root)
    route_ids = {route['id'] for route in reference['routes']}
    normal = index_attempts(run.get('routes', []), route_ids)
    pushes = index_attempts(run.get('disturbance', []), route_ids, 'force_N', reference['force_levels_N'])
    frictions = index_attempts(run.get('friction', []), route_ids, 'friction', reference['friction_levels'])
    used = set()

    def judge(route, attempt, field=None):
        path = contained(root, attempt.get('trace'))
        if path in used:
            raise FormatError('Each attempt needs its own state record')
        used.add(path)
        rows = load_trace(root, attempt['trace'])
        result = score_route(rows, route, reference['rules'], attempt['termination'])
        if field == 'force_N':
            record = read_json(contained(root, attempt.get('force_record')))
            valid = valid_force_record(rows, record, attempt[field], reference)
            result['force_record_verified'] = valid
            failure_reason = 'incomplete_disturbance'
        elif field == 'friction':
            valid = valid_friction_record(rows, attempt[field])
            result['friction_record_verified'] = valid
            failure_reason = 'friction_record_mismatch'
        else:
            return result
        if not valid:
            result['passed'] = False
            if result['reason'] == 'completed':
                result['reason'] = failure_reason
        return result

    route_results = []
    push_results = []
    friction_results = []
    for route in reference['routes']:
        route_id = route['id']
        if route_id in normal:
            route_results.append(judge(route, normal[route_id]))
        else:
            route_results.append({'route_id': route_id, 'score': 0, 'maximum': 6, 'reason': 'missing'})
        for field, levels, attempts, results in [
            ('force_N', reference['force_levels_N'], pushes, push_results),
            ('friction', reference['friction_levels'], frictions, friction_results),
        ]:
            for level in levels:
                attempt = attempts.get((route_id, level))
                result = judge(route, attempt, field) if attempt else None
                results.append({'route_id': route_id, field: level, 'maximum': .5,
                                'score': .5 if result and result['passed'] else 0,
                                'reason': result['reason'] if result else 'missing', 'attempt': result})

    video = {'score': 0, 'maximum': 20, 'reason': 'missing_video_predictions'}
    if video_reference is not None and run.get('video_predictions'):
        predictions = read_json(contained(root, run['video_predictions']))
        video = score_video(predictions, video_reference, include_answers=run.get('video_split') == 'train')

    route_score = sum(r['score'] for r in route_results)
    push_score = sum(r['score'] for r in push_results)
    friction_score = sum(r['score'] for r in friction_results)
    navigation_score = route_score + push_score + friction_score
    report = {
        'reference_id': reference['reference_id'], 'team': run.get('team', ''),
        'scores': {
            'route': {'score': route_score, 'maximum': 30},
            'disturbance': {'score': push_score + friction_score, 'maximum': 20,
                            'push': {'score': push_score, 'maximum': 10},
                            'friction': {'score': friction_score, 'maximum': 10}},
            'recognition': video,
        },
        'navigation_control_score': navigation_score, 'navigation_control_maximum': 50,
        'automated_score': navigation_score + video['score'], 'automated_maximum': 70,
        'maximum_score': 100, 'pending_manual': {'report': 30}, 'final_score': None,
        'details': {'routes': route_results, 'disturbance': push_results, 'friction': friction_results},
    }
    if 'sensors' in run:
        sensors = run['sensors']
        report['sensors'] = sensors
        report['sensor_mode'] = 'fusion' if len(sensors) == 2 else 'vision' if 'front_rgb' in sensors else 'lidar'
    return report


def add_report_score(report,score):
    if not isinstance(report,dict):
        raise FormatError('需要测评成绩文件')
    score=number(score,'报告分',0)
    if score>30:raise FormatError('报告分应在 0 到 30 之间')
    automated=number(report.get('automated_score'),'自动评分',0)
    if automated>70:raise FormatError('自动评分超出 70 分')
    result=copy.deepcopy(report)
    result.setdefault('scores',{})['report']={'score':score,'maximum':30}
    result['pending_manual']={}
    result['final_score']=automated+score
    result['maximum_score']=100
    return result
