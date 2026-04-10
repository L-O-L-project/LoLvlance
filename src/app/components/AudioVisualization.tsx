import { useEffect, useState } from 'react';
import { motion } from 'motion/react';

interface AudioVisualizationProps {
  state: 'idle' | 'listening' | 'result' | 'monitoring';
}

export function AudioVisualization({ state }: AudioVisualizationProps) {
  const [bars, setBars] = useState<number[]>(Array(5).fill(0.3));

  useEffect(() => {
    if (state === 'listening' || state === 'monitoring') {
      const interval = setInterval(() => {
        setBars(Array(5).fill(0).map(() => Math.random() * 0.7 + 0.3));
      }, 100);
      return () => clearInterval(interval);
    } else {
      setBars(Array(5).fill(0.3));
    }
  }, [state]);

  const isActive = state === 'listening' || state === 'monitoring';
  const glowColor = state === 'monitoring' ? 'bg-green-500/20' : 'bg-purple-500/20';
  const barColor = state === 'monitoring' 
    ? 'bg-gradient-to-t from-green-500 to-emerald-400' 
    : 'bg-gradient-to-t from-purple-500 to-blue-400';

  return (
    <div className="w-full flex items-center justify-center">
      <div className="relative">
        {/* Background Glow */}
        {isActive && (
          <motion.div
            className={`absolute inset-0 ${glowColor} blur-3xl rounded-full`}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.5, 0.3]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        )}
        
        {/* Visualization Bars */}
        <div className="relative flex items-end justify-center gap-3 h-48">
          {bars.map((height, index) => (
            <motion.div
              key={index}
              className={`w-3 rounded-full ${
                isActive 
                  ? barColor
                  : 'bg-gray-700'
              }`}
              animate={{
                height: `${height * 100}%`,
              }}
              transition={{
                duration: 0.1,
                ease: "easeOut"
              }}
              style={{
                minHeight: '20%'
              }}
            />
          ))}
        </div>

        {/* Circular Ring */}
        <motion.div
          className="absolute -inset-16 rounded-full border-2 border-gray-800"
          animate={isActive ? {
            scale: [1, 1.1, 1],
            opacity: [0.3, 0.6, 0.3]
          } : {}}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </div>
    </div>
  );
}