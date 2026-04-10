import { useState, useEffect, useRef } from 'react';
import type { AnalysisResult } from '../types';
import { generateMockProblems } from '../data/diagnosticData';

export function useMonitoring(onUpdate: (result: AnalysisResult) => void) {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const intervalRef = useRef<number | null>(null);

  const generateMockResult = (): AnalysisResult => {
    // 70% chance of detecting issues, 30% chance of no issues
    if (Math.random() > 0.3) {
      // 70% single problem, 20% two problems, 10% three problems
      const rand = Math.random();
      const problemCount = rand < 0.7 ? 1 : rand < 0.9 ? 2 : 3;
      
      return {
        problems: generateMockProblems(problemCount),
        timestamp: Date.now()
      };
    } else {
      return {
        problems: [],
        timestamp: Date.now()
      };
    }
  };

  const startMonitoring = () => {
    setIsMonitoring(true);
    
    // Initial update
    onUpdate(generateMockResult());
    
    // Update every 4 seconds
    intervalRef.current = window.setInterval(() => {
      onUpdate(generateMockResult());
    }, 4000);
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return {
    isMonitoring,
    startMonitoring,
    stopMonitoring
  };
}