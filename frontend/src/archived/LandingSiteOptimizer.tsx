import { AlertTriangle, Brain, CheckCircle, Star, Target, Zap } from 'lucide-react';
import React, { useState } from 'react';

interface OptimizationCriteria {
  scientific_value: number;
  safety_score: number;
  accessibility: number;
  resource_availability: number;
  communication_range: number;
  terrain_difficulty: number;
}

interface LandingSiteCandidate {
  id: string;
  name: string;
  coordinates: {
    latitude: number;
    longitude: number;
    elevation: number;
  };
  overall_score: number;
  criteria_scores: OptimizationCriteria;
  risk_factors: string[];
  opportunities: string[];
  ai_confidence: number;
  mission_suitability: {
    exploration: number;
    sample_return: number;
    human_precursor: number;
  };
}

interface Props {
  onSiteSelected?: (site: LandingSiteCandidate) => void;
  searchRegion?: {
    minLat: number;
    maxLat: number;
    minLon: number;
    maxLon: number;
  };
}

const LandingSiteOptimizer: React.FC<Props> = ({ onSiteSelected, searchRegion }) => {
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [candidates, setCandidates] = useState<LandingSiteCandidate[]>([]);
  const [selectedMissionType, setSelectedMissionType] = useState<'exploration' | 'sample_return' | 'human_precursor'>('exploration');
  const [criteriaWeights, setCriteriaWeights] = useState<OptimizationCriteria>({
    scientific_value: 0.25,
    safety_score: 0.30,
    accessibility: 0.20,
    resource_availability: 0.10,
    communication_range: 0.10,
    terrain_difficulty: 0.05
  });
  const [optimizationResults, setOptimizationResults] = useState<any>(null);

  // Sample landing site data
  const generateCandidates = async (): Promise<LandingSiteCandidate[]> => {
    // Simulate AI optimization process
    await new Promise(resolve => setTimeout(resolve, 2000));

    return [
      {
        id: 'opt-001',
        name: 'Delta Formation Alpha',
        coordinates: { latitude: 18.85, longitude: 77.52, elevation: -2540 },
        overall_score: 0.94,
        criteria_scores: {
          scientific_value: 0.95,
          safety_score: 0.88,
          accessibility: 0.92,
          resource_availability: 0.85,
          communication_range: 0.90,
          terrain_difficulty: 0.78
        },
        risk_factors: ['Seasonal dust storms', 'Moderate terrain slope'],
        opportunities: ['Ancient water activity', 'Organic preservation potential', 'Diverse mineralogy'],
        ai_confidence: 0.96,
        mission_suitability: {
          exploration: 0.94,
          sample_return: 0.91,
          human_precursor: 0.76
        }
      },
      {
        id: 'opt-002',
        name: 'Crater Rim Outpost',
        coordinates: { latitude: -5.4, longitude: 137.8, elevation: -4500 },
        overall_score: 0.87,
        criteria_scores: {
          scientific_value: 0.82,
          safety_score: 0.95,
          accessibility: 0.88,
          resource_availability: 0.70,
          communication_range: 0.85,
          terrain_difficulty: 0.92
        },
        risk_factors: ['Limited water access', 'Rocky terrain'],
        opportunities: ['Excellent safety record', 'Clear landing zone', 'High elevation'],
        ai_confidence: 0.89,
        mission_suitability: {
          exploration: 0.87,
          sample_return: 0.84,
          human_precursor: 0.91
        }
      },
      {
        id: 'opt-003',
        name: 'Valley Network Beta',
        coordinates: { latitude: -14.0, longitude: -59.2, elevation: -7000 },
        overall_score: 0.76,
        criteria_scores: {
          scientific_value: 0.98,
          safety_score: 0.55,
          accessibility: 0.65,
          resource_availability: 0.92,
          communication_range: 0.70,
          terrain_difficulty: 0.45
        },
        risk_factors: ['Steep canyon walls', 'Challenging access', 'Communication shadowing'],
        opportunities: ['Spectacular geology', 'Water flow evidence', 'Unique stratigraphy'],
        ai_confidence: 0.73,
        mission_suitability: {
          exploration: 0.85,
          sample_return: 0.78,
          human_precursor: 0.45
        }
      }
    ];
  };

  const runOptimization = async () => {
    setIsOptimizing(true);
    setOptimizationResults(null);

    try {
      // Simulate API call to backend optimization service
      const newCandidates = await generateCandidates();
      setCandidates(newCandidates);

      // Generate optimization summary
      const results = {
        total_candidates: newCandidates.length,
        search_region: searchRegion || { minLat: -90, maxLat: 90, minLon: -180, maxLon: 180 },
        best_candidate: newCandidates[0],
        criteria_analysis: {
          highest_scientific: newCandidates.reduce((prev, current) =>
            prev.criteria_scores.scientific_value > current.criteria_scores.scientific_value ? prev : current
          ),
          safest_option: newCandidates.reduce((prev, current) =>
            prev.criteria_scores.safety_score > current.criteria_scores.safety_score ? prev : current
          )
        },
        mission_recommendations: {
          exploration: newCandidates.filter(c => c.mission_suitability.exploration > 0.8).length,
          sample_return: newCandidates.filter(c => c.mission_suitability.sample_return > 0.8).length,
          human_precursor: newCandidates.filter(c => c.mission_suitability.human_precursor > 0.8).length
        }
      };

      setOptimizationResults(results);
    } catch (error) {
      // Handle optimization error silently
      setOptimizationResults(null);
    } finally {
      setIsOptimizing(false);
    }
  };

  const ScoreBar = ({ label, value, color = 'blue' }: { label: string; value: number; color?: string }) => (
    <div className="mb-2">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-600">{label}</span>
        <span className="font-medium">{Math.round(value * 100)}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`bg-${color}-500 h-2 rounded-full transition-all duration-300`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold flex items-center">
          <Target className="mr-3 text-blue-600" size={28} />
          AI-Powered Landing Site Optimization
        </h2>

        <div className="flex items-center space-x-2">
          <select
            value={selectedMissionType}
            onChange={(e) => setSelectedMissionType(e.target.value as any)}
            className="px-3 py-2 border rounded-md"
          >
            <option value="exploration">Exploration Mission</option>
            <option value="sample_return">Sample Return</option>
            <option value="human_precursor">Human Precursor</option>
          </select>

          <button
            onClick={runOptimization}
            disabled={isOptimizing}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center"
          >
            {isOptimizing ? (
              <>
                <Brain className="animate-pulse mr-2" size={18} />
                Optimizing...
              </>
            ) : (
              <>
                <Zap className="mr-2" size={18} />
                Run AI Optimization
              </>
            )}
          </button>
        </div>
      </div>

      {/* Criteria Weights Configuration */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Optimization Criteria Weights</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {Object.entries(criteriaWeights).map(([key, value]) => (
            <div key={key}>
              <label className="block text-sm text-gray-600 mb-1 capitalize">
                {key.replace('_', ' ')}: {Math.round(value * 100)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={value}
                onChange={(e) => {
                  setCriteriaWeights(prev => ({
                    ...prev,
                    [key]: parseFloat(e.target.value)
                  }));
                }}
                className="w-full"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Optimization Results Summary */}
      {optimizationResults && (
        <div className="mb-6 p-4 bg-green-50 rounded-lg border border-green-200">
          <h3 className="text-lg font-semibold text-green-800 mb-3 flex items-center">
            <CheckCircle className="mr-2" size={20} />
            Optimization Complete
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="font-medium text-green-700">Best Overall</div>
              <div>{optimizationResults.best_candidate?.name}</div>
              <div className="text-green-600">
                Score: {Math.round(optimizationResults.best_candidate?.overall_score * 100)}%
              </div>
            </div>

            <div>
              <div className="font-medium text-green-700">Highest Science Value</div>
              <div>{optimizationResults.criteria_analysis?.highest_scientific?.name}</div>
              <div className="text-green-600">
                Science: {Math.round(optimizationResults.criteria_analysis?.highest_scientific?.criteria_scores.scientific_value * 100)}%
              </div>
            </div>

            <div>
              <div className="font-medium text-green-700">Safest Option</div>
              <div>{optimizationResults.criteria_analysis?.safest_option?.name}</div>
              <div className="text-green-600">
                Safety: {Math.round(optimizationResults.criteria_analysis?.safest_option?.criteria_scores.safety_score * 100)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Landing Site Candidates */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Optimized Landing Site Candidates</h3>

        {candidates.length === 0 && !isOptimizing && (
          <div className="text-center py-8 text-gray-500">
            Click "Run AI Optimization" to generate landing site recommendations
          </div>
        )}

        {isOptimizing && (
          <div className="text-center py-8">
            <Brain className="animate-pulse mx-auto mb-4 text-blue-600" size={48} />
            <div className="text-lg font-medium">AI Optimization in Progress...</div>
            <div className="text-sm text-gray-600">
              Analyzing terrain, safety factors, and scientific potential
            </div>
          </div>
        )}

        {candidates.map((candidate, index) => (
          <div
            key={candidate.id}
            className="border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
            onClick={() => onSiteSelected?.(candidate)}
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <h4 className="text-lg font-semibold flex items-center">
                  {index === 0 && <Star className="text-yellow-500 mr-2" size={20} />}
                  {candidate.name}
                </h4>
                <div className="text-sm text-gray-600">
                  {candidate.coordinates.latitude.toFixed(3)}°, {candidate.coordinates.longitude.toFixed(3)}°
                  (Elevation: {candidate.coordinates.elevation}m)
                </div>
              </div>

              <div className="text-right">
                <div className="text-2xl font-bold text-blue-600">
                  {Math.round(candidate.overall_score * 100)}%
                </div>
                <div className="text-xs text-gray-500">Overall Score</div>
                <div className="text-xs text-gray-500">
                  AI Confidence: {Math.round(candidate.ai_confidence * 100)}%
                </div>
              </div>
            </div>

            {/* Mission Suitability */}
            <div className="mb-3">
              <div className="text-sm font-medium mb-2">Mission Suitability</div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="text-center">
                  <div className="font-medium">Exploration</div>
                  <div className={`text-lg ${candidate.mission_suitability.exploration > 0.8 ? 'text-green-600' : 'text-yellow-600'}`}>
                    {Math.round(candidate.mission_suitability.exploration * 100)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="font-medium">Sample Return</div>
                  <div className={`text-lg ${candidate.mission_suitability.sample_return > 0.8 ? 'text-green-600' : 'text-yellow-600'}`}>
                    {Math.round(candidate.mission_suitability.sample_return * 100)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="font-medium">Human Precursor</div>
                  <div className={`text-lg ${candidate.mission_suitability.human_precursor > 0.8 ? 'text-green-600' : 'text-yellow-600'}`}>
                    {Math.round(candidate.mission_suitability.human_precursor * 100)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Detailed Criteria Scores */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
              <div>
                <ScoreBar label="Scientific Value" value={candidate.criteria_scores.scientific_value} color="green" />
                <ScoreBar label="Safety Score" value={candidate.criteria_scores.safety_score} color="blue" />
                <ScoreBar label="Accessibility" value={candidate.criteria_scores.accessibility} color="purple" />
              </div>
              <div>
                <ScoreBar label="Resources" value={candidate.criteria_scores.resource_availability} color="yellow" />
                <ScoreBar label="Communication" value={candidate.criteria_scores.communication_range} color="indigo" />
                <ScoreBar label="Terrain Ease" value={candidate.criteria_scores.terrain_difficulty} color="gray" />
              </div>
            </div>

            {/* Risk Factors and Opportunities */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <div className="flex items-center text-sm font-medium text-red-700 mb-1">
                  <AlertTriangle size={16} className="mr-1" />
                  Risk Factors
                </div>
                <ul className="text-xs text-red-600 list-disc list-inside">
                  {candidate.risk_factors.map((risk, idx) => (
                    <li key={idx}>{risk}</li>
                  ))}
                </ul>
              </div>

              <div>
                <div className="flex items-center text-sm font-medium text-green-700 mb-1">
                  <CheckCircle size={16} className="mr-1" />
                  Opportunities
                </div>
                <ul className="text-xs text-green-600 list-disc list-inside">
                  {candidate.opportunities.map((opportunity, idx) => (
                    <li key={idx}>{opportunity}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LandingSiteOptimizer;
